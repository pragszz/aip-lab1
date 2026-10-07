#!/usr/bin/env python3
"""Build the cache that CI replays.

The working cache (.aip_cache/calls.sqlite3) had grown to 147 MB across Labs
1-6, over GitHub's 100 MB per-file limit, and almost none of it is needed by
the gate. This runs any command in-process while recording every cache key it
reads or writes, then copies exactly those rows into a separate SQLite file.

    # 1. run the gate (online, or against the full cache) and record its keys
    python labs/lab7/ci_cache.py record --keys reports/lab7_cache_keys.txt -- \
        labs/lab7/gate.py --config labs/lab7/thresholds.yml

    # 2. copy those rows out of the full cache into the one we commit
    python labs/lab7/ci_cache.py export --keys reports/lab7_cache_keys.txt \
        --src .aip_cache_full/calls.sqlite3 --dst .aip_cache/calls.sqlite3

Key files are appended to, so recording several runs (the gate, the D3 break,
the demo questions) unions them.
"""
from __future__ import annotations

import argparse
import runpy
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def record(keys_path: Path, argv: list[str]) -> int:
    from aip import cache

    seen: set[str] = set()
    real_get, real_put = cache.get, cache.put

    def get(key):
        seen.add(key)
        return real_get(key)

    def put(key, kind, request, response):
        seen.add(key)
        return real_put(key, kind, request, response)

    cache.get, cache.put = get, put
    sys.argv = argv
    code = 0
    try:
        runpy.run_path(str(ROOT / argv[0]), run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code or 0)
    finally:
        keys_path.parent.mkdir(parents=True, exist_ok=True)
        with keys_path.open("a", encoding="utf-8") as fh:
            fh.writelines(k + "\n" for k in sorted(seen))
        print(f"recorded {len(seen)} keys -> {keys_path}", file=sys.stderr)
    return code


def export(keys_path: Path, src: Path, dst: Path) -> None:
    keys = sorted({line.strip() for line in keys_path.open() if line.strip()})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = sqlite3.connect(dst)
    out.execute("""CREATE TABLE IF NOT EXISTS calls (
                   key TEXT PRIMARY KEY, kind TEXT NOT NULL, request TEXT NOT NULL,
                   response TEXT NOT NULL, created_at REAL NOT NULL)""")
    inp = sqlite3.connect(src)
    copied = 0
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        rows = inp.execute(
            f"SELECT key, kind, request, response, created_at FROM calls "
            f"WHERE key IN ({','.join('?' * len(part))})", part).fetchall()
        out.executemany("INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?)", rows)
        copied += len(rows)
    out.commit()
    out.execute("VACUUM")
    out.close()
    print(f"copied {copied}/{len(keys)} keys into {dst} "
          f"({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--keys", type=Path, required=True)
    r.add_argument("script", nargs=argparse.REMAINDER)
    e = sub.add_parser("export")
    e.add_argument("--keys", type=Path, required=True)
    e.add_argument("--src", type=Path, required=True)
    e.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "record":
        argv = a.script[1:] if a.script and a.script[0] == "--" else a.script
        return record(a.keys, argv)
    export(a.keys, a.src, a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
