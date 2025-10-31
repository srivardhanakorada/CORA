#!/usr/bin/env python3
"""
Generate GenBench-40 CSV from data.py

Expects data.py to define:
  - identity_templates: List[str]   # 30 templates for celebrity identities
  - ip_templates:        List[str]   # 30 templates for IP characters/logos
  - targets: List[dict]              # {category, target_name, variants}

Output CSV columns:
  case_id, seed, prompt, target_name, type, template_id, category, paraphrase_id
"""
import argparse
import csv
import hashlib
import importlib.util
import os
import sys
from typing import List, Dict


def load_data_module(path: str):
    spec = importlib.util.spec_from_file_location("data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det_seed(*parts: str, mod: int = 10**9 + 7) -> int:
    """
    Deterministic seed from arbitrary strings (stable across runs/machines).
    """
    s = "|".join(parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def sanitize_text(s: str) -> str:
    """
    Minimal normalization to avoid hidden whitespace issues.
    (Do NOT over-normalize; we keep user intent intact.)
    """
    return " ".join(s.split())


def validate_templates(name: str, tmpls: List[str]) -> None:
    if len(tmpls) != 30:
        raise ValueError(f"{name} must have exactly 30 templates (got {len(tmpls)}).")
    # Ensure each template has exactly one {} placeholder
    for i, t in enumerate(tmpls):
        if t.count("{}") != 1:
            raise ValueError(f"{name}[{i}] must contain exactly one '{{}}' placeholder. Got: {t}")


def build_rows(
    identity_templates: List[str],
    ip_templates: List[str],
    targets: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    validate_templates("identity_templates", identity_templates)
    validate_templates("ip_templates", ip_templates)

    rows: List[Dict[str, str]] = []
    case_id = 0

    for tgt in targets:
        category = sanitize_text(tgt["category"])
        target_name = sanitize_text(tgt["target_name"])

        # choose the right template list
        if category == "celebrity":
            templates = identity_templates
        elif category == "ip":
            templates = ip_templates
        else:
            raise ValueError(f"Unknown category '{category}' for target '{target_name}'")

        # Build variant list: original (the literal target name) + provided variants
        raw_variants = [target_name] + [sanitize_text(v) for v in tgt.get("variants", [])]

        # Deduplicate variants while preserving order
        seen = set()
        variants = []
        for v in raw_variants:
            if v not in seen:
                seen.add(v)
                variants.append(v)

        # Expand to prompts
        for paraphrase_id, variant in enumerate(variants):
            vtype = "original" if paraphrase_id == 0 else "paraphrase"
            for template_id, tmpl in enumerate(templates):
                prompt = tmpl.format(variant)
                seed = det_seed(target_name, category, str(paraphrase_id), str(template_id))

                rows.append(
                    {
                        "case_id": str(case_id),
                        "seed": str(seed),
                        "prompt": prompt,
                        "target_name": target_name,
                        "type": vtype,
                        "template_id": str(template_id),
                        "category": category,
                        "paraphrase_id": str(paraphrase_id),
                    }
                )
            case_id += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Create GenBench-40 CSV from data.py")
    parser.add_argument(
        "--data",
        type=str,
        default="data.py",
        help="Path to data.py containing identity_templates, ip_templates, targets",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="genbench40.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    data = load_data_module(args.data)

    # Basic presence checks
    required = ["identity_templates", "ip_templates", "targets"]
    for attr in required:
        if not hasattr(data, attr):
            print(f"ERROR: '{attr}' missing in {args.data}", file=sys.stderr)
            sys.exit(1)

    rows = build_rows(
        identity_templates=getattr(data, "identity_templates"),
        ip_templates=getattr(data, "ip_templates"),
        targets=getattr(data, "targets"),
    )

    # Write CSV
    fieldnames = [
        "case_id",
        "seed",
        "prompt",
        "target_name",
        "type",
        "template_id",
        "category",
        "paraphrase_id",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
