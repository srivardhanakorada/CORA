import os
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="results/adavd/genbench")
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--erase_type", type=str, default="celebs")
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sigmoid_a", type=float, default=100)
    parser.add_argument("--sigmoid_b", type=float, default=0.93)
    parser.add_argument("--sigmoid_c", type=float, default=2)
    parser.add_argument("--decomp_timestep", type=int, default=0)
    parser.add_argument("--record_type", type=str, default="values")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["target_name"] = df["target_name"].str.strip()

    print(f"[INFO] Running AdaVD for {len(df)} GenBench rows...")
    os.makedirs(args.save_root, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GenBench Rows"):
        target = row["target_name"]
        prompt = row["prompt"]
        seed = int(row["seed"])

        cmd = [
            "python", "adavd_src/main_adavd.py",
            "--sd_ckpt", args.sd_ckpt,
            "--seed", str(seed),
            "--mode", "original,retain",
            "--erase_type", args.erase_type,
            "--target_concept", target,
            "--contents", prompt,
            "--total_timesteps", str(args.total_timesteps),
            "--guidance_scale", str(args.guidance_scale),
            "--sigmoid_a", str(args.sigmoid_a),
            "--sigmoid_b", str(args.sigmoid_b),
            "--sigmoid_c", str(args.sigmoid_c),
            "--record_type", args.record_type,
            "--batch_size", "1",
            "--num_samples", "1",
            "--save_path", args.save_root,
        ]

        print(f"\n[Row {idx}] {target} | Seed={seed}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
