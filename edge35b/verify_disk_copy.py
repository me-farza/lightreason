"""Audit a copied checkpoint before optionally replacing its RAM-backed shards with links."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from edge35b.download import REVISION


def identity(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--allow-ram-dir", type=Path, action="append", default=[])
    p.add_argument("--link-verified", action="store_true")
    args = p.parse_args()
    source, copy = args.source.resolve(), args.copy.resolve()
    if source == copy:
        p.error("Source and copy must be distinct directories")
    manifest_file = "lightreason-download.json"
    manifest = json.loads((source / manifest_file).read_bytes())
    if manifest["revision"] != REVISION or (source / manifest_file).read_bytes() != (copy / manifest_file).read_bytes():
        raise ValueError("Wrong checkpoint manifest")
    index = "model.safetensors.index.json"
    if (source / index).read_bytes() != (copy / index).read_bytes():
        raise ValueError("Shard index changed in copy")
    names = sorted(set(json.loads((source / index).read_bytes())["weight_map"].values()))
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("Invalid index")
    digests = {item["rfilename"]: item["lfs"]["sha256"] for item in manifest["files"] if item.get("lfs")}
    if args.link_verified:
        audit = json.loads(args.audit.read_bytes())
        rows = audit["shards"]
        if audit["revision"] != REVISION or audit["copy"] != str(copy) or set(rows) != set(names):
            raise ValueError("Audit does not cover the requested copy")
        allowed = {d.resolve() for d in args.allow_ram_dir}
        pending = []
        for name in names:
            disk = copy / name
            origin = (source / name).resolve()
            row = rows[name]
            if disk.is_symlink() or identity(disk) != row["identity"] or row["sha256"] != digests[name]:
                raise ValueError(f"Copied shard changed since verification: {name}")
            if origin == disk:
                continue
            if origin.parent not in allowed or str(origin) != row["origin"] or not origin.is_file():
                raise ValueError(f"Unexpected original shard path: {origin}")
            pending.append((origin, disk))
        for origin, disk in pending:
            temp = origin.with_name(f".{origin.name}.{os.getpid()}.verified-link")
            try:
                temp.symlink_to(disk)
                os.replace(temp, origin)
            finally:
                temp.unlink(missing_ok=True)
        print(f"Replaced {len(pending)} audited shard paths with links to the same checkpoint.", flush=True)
        return

    rows = {}
    for name in names:
        disk = copy / name
        origin = (source / name).resolve()
        before = identity(disk)
        if disk.is_symlink() or before[2] != origin.stat().st_size:
            raise ValueError(f"Unexpected disk copy: {disk}")
        with disk.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != digests[name] or before != identity(disk):
            raise ValueError(f"Shard failed pinned checksum: {name}")
        rows[name] = {"identity": before, "sha256": digest, "origin": str(origin)}
        print(f"Verified {name}", flush=True)
    temp = args.audit.with_suffix(".part")
    temp.write_text(json.dumps({"revision": REVISION, "copy": str(copy), "shards": rows}, indent=2) + "\n")
    temp.replace(args.audit)


if __name__ == "__main__":
    main()
