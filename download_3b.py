"""Download the pinned public checkpoint directly; verify official object hashes.

No model conversion, cache duplication, credentials, or GPU work. Destination
must have enough space; partial files can resume and are atomically renamed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request


def hashes(path, size):
    sha = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    if path.exists():
        with path.open("rb") as source:
            while data := source.read(8 * 1024 * 1024):
                sha.update(data)
                git.update(data)
    return sha, git


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="results3b/download-plan.json")
    p.add_argument("--destination", required=True)
    p.add_argument("--out", default="results3b/model-manifest.json")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.destination)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = {"model_id": plan["model_id"], "revision": plan["revision"],
              "destination": str(root.resolve()), "files": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for entry in plan["files"]:
        name, size = entry["rfilename"], entry["size"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid repository filename")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("sha256", entry["blobId"])
        kind = "sha256" if "lfs" in entry else "git_blob_sha1"
        if path.exists():
            sha, git = hashes(path, size)
            actual = sha.hexdigest() if "lfs" in entry else git.hexdigest()
            if path.stat().st_size != size or actual != expected:
                raise RuntimeError(f"Existing complete file fails official hash: {name}")
        else:
            partial = path.with_name(path.name + ".part")
            for attempt in range(1, 5):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size:
                    raise RuntimeError(f"Oversized partial file: {name}")
                if shutil.disk_usage(root).free < size - offset + 2 * 1024 ** 3:
                    raise RuntimeError("Insufficient free space including2GiB reserve")
                sha, git = hashes(partial, size)
                if offset == size:
                    break
                url = ("https://huggingface.co/" + plan["model_id"] + "/resolve/" +
                       plan["revision"] + "/" + urllib.parse.quote(name) + "?download=true")
                headers = {"User-Agent": "Lightreason-exact-benchmark/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                try:
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=90) as response:
                        if offset and response.status != 206:
                            offset = 0
                            partial.unlink()
                            sha, git = hashes(partial, size)
                        mode = "ab" if offset else "wb"
                        progress = offset
                        next_log = offset + 256 * 1024 ** 2
                        with partial.open(mode) as destination:
                            while data := response.read(8 * 1024 ** 2):
                                destination.write(data)
                                sha.update(data)
                                git.update(data)
                                progress += len(data)
                                if progress >= next_log:
                                    print(json.dumps({"file": name, "downloaded_bytes": progress,
                                                      "total_bytes": size}), flush=True)
                                    next_log = progress + 256 * 1024 ** 2
                    if partial.stat().st_size != size:
                        raise OSError("Incomplete HTTP response")
                    break
                except (OSError, urllib.error.URLError) as error:
                    print(json.dumps({"file": name, "attempt": attempt, "error": str(error)}), flush=True)
                    if attempt == 4:
                        raise
            actual = sha.hexdigest() if "lfs" in entry else git.hexdigest()
            if partial.stat().st_size != size or actual != expected:
                raise RuntimeError(f"Official checkpoint hash failed: {name}")
            partial.replace(path)
        result["files"][name] = {"bytes": size, "sha256": sha.hexdigest(),
                                   "official_object_hash_kind": kind,
                                   "official_object_hash": expected, "official_hash_verified": True}
        result["elapsed_s"] = time.time() - started
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({"file": name, "verified": True, "bytes": size}), flush=True)
    result["complete"] = True
    result["free_disk_bytes_after"] = shutil.disk_usage(root).free
    result["elapsed_s"] = time.time() - started
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"complete": True, "elapsed_s": result["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
