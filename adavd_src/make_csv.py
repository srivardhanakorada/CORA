#!/usr/bin/env python3
"""
Build a CSV with columns:
id,type,text,concept,seed
"""

import csv
import argparse
from pathlib import Path
from typing import List, Optional

def read_names(cli_list: List[str], file_path: Optional[str]) -> List[str]:
    names: List[str] = []
    if cli_list:
        names.extend(cli_list)
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {file_path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
    # Keep order but drop duplicates
    seen = set()
    deduped: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped

def main():
    ap = argparse.ArgumentParser(description="Generate erase/retain prompt CSV.")
    ap.add_argument("--erase", nargs="*", default=[], help="Names to label as type=erase.")
    ap.add_argument("--retain", nargs="*", default=[], help="Names to label as type=retain.")
    ap.add_argument("--erase-file", default=None, help="Text file with erase names (one per line).")
    ap.add_argument("--retain-file", default=None, help="Text file with retain names (one per line).")
    ap.add_argument("--seeds", type=int, default=10, help="Seeds per name (starts at 0).")
    ap.add_argument("--out", default="prompts.csv", help="Output CSV path.")
    ap.add_argument("--template", default="A photo of {name}.",
                    help="Prompt template; must contain {name}.")
    args = ap.parse_args()

    if "{name}" not in args.template:
        raise ValueError("--template must contain the placeholder {name}")

    erase_names  = read_names(args.erase,  args.erase_file)
    retain_names = read_names(args.retain, args.retain_file)

    rows = []
    idx = 0

    def add_rows(name_list: List[str], typ: str):
        nonlocal idx, rows
        for name in name_list:
            for s in range(args.seeds):
                text = args.template.format(name=name)
                rows.append([idx, typ, text, name, s])
                idx += 1

    add_rows(erase_names,  "erase")
    add_rows(retain_names, "retain")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "text", "concept", "seed"])
        w.writerows(rows)

    print(f"✅ Wrote {len(rows)} rows to {args.out} "
          f"(erase={len(erase_names)} names, retain={len(retain_names)} names, seeds={args.seeds}).")

if __name__ == "__main__":
    main()
