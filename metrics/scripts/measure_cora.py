#!/usr/bin/env python3
import argparse, subprocess, os, sys

def run_eval(esr_script, folder, targets, non_targets, out_csv, model, device):
    cmd = [
        sys.executable, esr_script,
        "--folder", folder,
        "--targets", targets,
        "--non_targets", non_targets,
        "--csv_out", out_csv,
        "--model", model,
        "--device", device,
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Unified ESR/PSR/HM runner for CoRA (original + erase).")
    ap.add_argument("--esr_script", default="metrics/src/cs_metrics.py",
                    help="Path to your ORIGINAL ESR/PSR/HM script (kept unchanged).")
    ap.add_argument("--base_path", required=True,
                    help="Base results path for this CoRA run (contains 'original' and 'erase').")
    ap.add_argument("--target", required=True, help='Primary target concept, e.g. "Donald Trump"')
    ap.add_argument("--paraphrases", required=True,
                    help='Comma-separated paraphrases for the target (will ALWAYS be included).')
    ap.add_argument("--non_targets", required=True,
                    help='Comma-separated non-target concepts, e.g. "Lemon, Dog"')
    ap.add_argument("--model", default="ViT-L/14", help="CLIP model for esr_psr_hm.py")
    ap.add_argument("--device", default="cuda", help="Device for esr_psr_hm.py")
    args = ap.parse_args()

    # Always include paraphrases in the target set
    targets = f"{args.target}, {args.paraphrases}"

    # CoRA subfolders
    orig_folder  = os.path.join(args.base_path, "original")
    erase_folder = os.path.join(args.base_path, "erase")

    # Output CSVs
    orig_csv  = os.path.join(args.base_path, "original_esr_psr_hm.csv")
    erase_csv = os.path.join(args.base_path, "erase_esr_psr_hm.csv")

    # Run both modes
    run_eval(args.esr_script, orig_folder,  targets, args.non_targets, orig_csv,  args.model, args.device)
    run_eval(args.esr_script, erase_folder, targets, args.non_targets, erase_csv, args.model, args.device)

if __name__ == "__main__":
    main()
