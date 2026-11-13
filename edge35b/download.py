"""Stage and verify the pinned BF16 checkpoint across two scratch volumes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
HUB = "https://huggingface.co"
MIRROR = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3.6-35B-A3B/repo"
GIB = 1 << 30


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read host MemAvailable")


def file_hash(path, size, is_lfs):
    digest = hashlib.sha256() if is_lfs else hashlib.sha1(f"blob {size}\0".encode())
    if path.exists():
        with path.open("rb") as source:
            while chunk := source.read(8 << 20):
                digest.update(chunk)
    return digest


def fetch(entry, path, mirrored=False):
    name, size = entry["rfilename"], entry["size"]
    lfs = entry.get("lfs")
    expected = lfs["sha256"] if lfs else entry["blobId"]
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        if path.stat().st_size == size and file_hash(path, size, lfs).hexdigest() == expected:
            print(f"Verified existing {name}", flush=True)
            return
        raise ValueError(f"Existing file failed checkpoint verification: {path}")
    hub_url = f"{HUB}/{MODEL}/resolve/{REVISION}/{quote(name)}?download=true"
    urls = (f"{MIRROR}?Revision=master&FilePath={quote(name)}", hub_url) if mirrored else (hub_url,)
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Oversized partial download: {part}")
    digest = file_hash(part, size, lfs)
    attempt = 0
    while offset < size:
        before = offset
        try:
            end = min(offset + (64 << 20), size) - 1
            use_range = bool(lfs or offset)
            headers = {"Range": f"bytes={offset}-{end}"} if use_range else {}
            with urlopen(Request(urls[attempt % len(urls)], headers=headers), timeout=45) as source:
                content_range = source.headers.get("Content-Range")
                if use_range and (source.status != 206 or content_range != f"bytes {offset}-{end}/{size}"):
                    raise OSError(f"Server did not return the requested byte range for {name}")
                with part.open("ab", buffering=0) as output:
                    while chunk := source.read(8 << 20):
                        if available_memory() < 14 * GIB:
                            raise MemoryError("Stopped staging to preserve 14 GiB of host memory")
                        if offset + len(chunk) > end + 1:
                            raise ValueError(f"Oversized download response for {name}")
                        if output.write(chunk) != len(chunk):
                            raise OSError(f"Short scratch write for {name}")
                        digest.update(chunk)
                        offset += len(chunk)
            if offset != end + 1:
                raise OSError(f"Short range for {name}: {offset}/{end + 1}")
            attempt = 0
        except (OSError, HTTPException) as error:
            written = part.stat().st_size if part.exists() else 0
            if written != offset:
                offset = written
                digest = file_hash(part, size, lfs)
            attempt = 1 if offset > before else attempt + 1
            if attempt >= 16:
                raise
            print(f"Retry {name}: {error}", flush=True)
            time.sleep(min(20, 2 ** attempt))
    if part.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError(f"Upstream hash mismatch for {name}")
    part.replace(path)
    print(f"Verified {name} ({size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--primary-gib", type=int, default=54)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--mirror", action="store_true",
                        help="Try Qwen's official ModelScope mirror first for byte-identical shards")
    parser.add_argument("--overflow-dir", type=Path, help="Optional third scratch volume")
    parser.add_argument("--overflow-shard", action="append", default=[],
                        help="Shard filename to place on the third volume; repeat as needed")
    args = parser.parse_args()
    primary, secondary = args.model_dir.resolve(), args.shard_dir.resolve()
    overflow = args.overflow_dir.resolve() if args.overflow_dir else None
    if bool(overflow) != bool(args.overflow_shard):
        parser.error("Use --overflow-dir with one or more --overflow-shard options")
    for folder in {primary, secondary, overflow} - {None}:
        folder.mkdir(parents=True, exist_ok=True)
    with (primary / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage_checkpoint(primary, secondary, overflow, args)


def stage_checkpoint(primary, secondary, overflow, args):
    url = f"{HUB}/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Hugging Face returned an unexpected revision")
    wanted = (".safetensors", ".json", ".jinja", ".txt", "LICENSE")
    entries = [item for item in metadata["siblings"] if item["rfilename"].endswith(wanted)]
    shards = [item for item in entries if item["rfilename"].endswith(".safetensors")]
    if len(shards) != 26:
        raise ValueError("Unexpected checkpoint shard count")
    overflow_shards = set(args.overflow_shard)
    if overflow_shards - {item["rfilename"] for item in shards}:
        raise ValueError("An overflow file is outside the pinned shard list")
    mirrored = set()
    if args.mirror:
        with urlopen(f"{MIRROR}/files?Revision=master&Recursive=True", timeout=30) as response:
            mirror_files = {item["Path"]: (item["Size"], item.get("Sha256"))
                            for item in json.load(response)["Data"]["Files"]}
        for item in shards:
            name = item["rfilename"]
            if mirror_files.get(name) != (item["size"], item["lfs"]["sha256"]):
                raise ValueError(f"Official mirror does not match the pinned shard: {name}")
            mirrored.add(name)
    placement = []
    primary_bytes = 0
    pending = {}
    for entry in sorted(entries, key=lambda item: item["rfilename"].endswith(".safetensors")):
        name, size = entry["rfilename"], entry["size"]
        if Path(name).name != name:
            raise ValueError(f"Unexpected nested path: {name}")
        folder = primary if primary_bytes + size <= args.primary_gib * GIB else secondary
        if folder == primary:
            primary_bytes += size
        if name in overflow_shards:
            folder = overflow
        path = folder / name
        part = path.with_suffix(path.suffix + ".part")
        present = path.stat().st_size if path.exists() else part.stat().st_size if part.exists() else 0
        pending[folder] = pending.get(folder, 0) + max(0, size - present)
        placement.append((entry, path))
    by_volume = {}
    for folder, size in pending.items():
        device = folder.stat().st_dev
        mount_path, subtotal = by_volume.get(device, (folder, 0))
        by_volume[device] = (mount_path, subtotal + size)
    for folder, size in by_volume.values():
        if shutil.disk_usage(folder).free < size + 2 * GIB:
            raise OSError(f"Insufficient scratch space at {folder} including a 2 GiB reserve")
    if available_memory() < sum(pending.values()) + 16 * GIB:
        raise MemoryError("Insufficient available memory for scratch staging and a 16 GiB reserve")
    print(f"Downloading {sum(pending.values()) / GIB:.2f} GiB at revision {REVISION}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, entry, path, entry["rfilename"] in mirrored) for entry, path in placement]
        for future in futures:
            future.result()
    for entry, path in placement:
        link = primary / path.name
        if path.parent != primary:
            if link.is_symlink() and link.readlink() == path:
                continue
            link.symlink_to(path)
    manifest = {"model": MODEL, "revision": REVISION, "files": entries}
    staged_manifest = primary / "lightreason-download.json.part"
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    staged_manifest.replace(primary / "lightreason-download.json")
    print(f"Checkpoint ready: {primary}", flush=True)


if __name__ == "__main__":
    main()
