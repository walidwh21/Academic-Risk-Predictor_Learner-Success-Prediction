#!/usr/bin/env python3
"""Command-line front end for the student record registry.

    python manage_records.py add --name "Sam Lee" --age 17 \
        --track Science --email sam@example.com --courses "Math, Physics"
    python manage_records.py list
    python manage_records.py search Science
    python manage_records.py show S0001
    python manage_records.py update S0001 --age 18
    python manage_records.py courses S0001 --add Chemistry --remove Physics
    python manage_records.py delete S0001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_model.records import (Registry, ValidationError,  # noqa: E402
                                split_courses)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Student record registry")
    p.add_argument("--db", default="students.db", help="SQLite file to use")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="create a record")
    a.add_argument("--name", required=True)
    a.add_argument("--age", required=True)
    a.add_argument("--track", default="")
    a.add_argument("--email", required=True)
    a.add_argument("--courses", default="", help="comma separated")

    sub.add_parser("list", help="show every record")

    s = sub.add_parser("search", help="match name, track or course")
    s.add_argument("text")

    sh = sub.add_parser("show", help="show one record")
    sh.add_argument("id")

    u = sub.add_parser("update", help="change basic fields")
    u.add_argument("id")
    for f in ("name", "age", "track", "email"):
        u.add_argument(f"--{f}")

    c = sub.add_parser("courses", help="add or remove courses")
    c.add_argument("id")
    c.add_argument("--add", default="")
    c.add_argument("--remove", default="")

    d = sub.add_parser("delete", help="remove a record")
    d.add_argument("id")
    return p


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reg = Registry(args.db)
    try:
        if args.cmd == "add":
            rec = reg.add(args.name, args.age, args.track, args.email,
                          split_courses(args.courses))
            print(f"Created {rec.record_id}\n{rec}")
        elif args.cmd == "list":
            rows = reg.all()
            print("\n".join(map(str, rows)) if rows else "No records yet.")
            print(f"\n{len(rows)} record(s)")
        elif args.cmd == "search":
            rows = reg.search(args.text)
            print("\n".join(map(str, rows)) if rows else "No matches.")
        elif args.cmd == "show":
            rec = reg.get(args.id)
            print(rec if rec else f"{args.id} not found")
            return 0 if rec else 1
        elif args.cmd == "update":
            changes = {k: getattr(args, k) for k in ("name", "age", "track", "email")
                       if getattr(args, k) is not None}
            rec = reg.update(args.id, **changes)
            print(f"Updated\n{rec}" if rec else f"{args.id} not found")
            return 0 if rec else 1
        elif args.cmd == "courses":
            rec = reg.set_courses(args.id, split_courses(args.add),
                                  split_courses(args.remove))
            print(f"Updated\n{rec}" if rec else f"{args.id} not found")
            return 0 if rec else 1
        elif args.cmd == "delete":
            ok = reg.delete(args.id)
            print("Deleted" if ok else f"{args.id} not found")
            return 0 if ok else 1
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        reg.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
