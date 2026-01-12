#!/usr/bin/env python3
# collect_elm_artifacts.py
# Copie:
#  - eval_in_dist/eval_results_mlm.txt et eval_out_dist/eval_results_mlm.txt
#    pour chaque model-* de chaque run
#  - train_total_loss.csv et train_env_losses.csv à la racine de chaque run

import argparse
from pathlib import Path
import shutil

TRAIN_FILES = ["train_total_loss.csv", "train_env_losses.csv"]
EVAL_SPLITS = ["eval_in_dist", "eval_out_dist"]
EVAL_FILENAME = "eval_results_mlm.txt"

def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def collect_eval(src_root: Path, dst_root: Path):
    copied, missing = 0, []
    for run_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in run_dir.iterdir()
                                if p.is_dir() and p.name.startswith("model-")):
            for split in EVAL_SPLITS:
                src_file = model_dir / split / EVAL_FILENAME
                if src_file.is_file():
                    dst_file = dst_root / run_dir.name / model_dir.name / split / EVAL_FILENAME
                    _copy(src_file, dst_file)
                    copied += 1
                else:
                    missing.append(src_file)
    return copied, missing

def collect_train_csvs(src_root: Path, dst_root: Path):
    copied, missing = 0, []
    for run_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for fname in TRAIN_FILES:
            src_file = run_dir / fname
            if src_file.is_file():
                dst_file = dst_root / run_dir.name / fname
                _copy(src_file, dst_file)
                copied += 1
            else:
                missing.append(src_file)
    return copied, missing

def main():
    parser = argparse.ArgumentParser(description="Collecter eval (in/out) et CSV d'entraînement depuis runs_elm.")
    parser.add_argument("--src", type=Path, required=True, help="Dossier source (ex: runs_elm)")
    parser.add_argument("--dst-eval", type=Path, default=Path("saved_eval_elm"),
                        help="Dossier destination pour les évaluations")
    parser.add_argument("--dst-train", type=Path, default=Path("saved_train_elm"),
                        help="Dossier destination pour les CSV d'entraînement")
    parser.add_argument("--only", choices=["both", "eval", "train"], default="both",
                        help="Que collecter (par défaut: both)")
    args = parser.parse_args()

    if args.only in ("both", "eval"):
        args.dst_eval.mkdir(parents=True, exist_ok=True)
        n, miss = collect_eval(args.src, args.dst_eval)
        print(f"[EVAL] Fichiers copiés : {n}")
        if miss:
            print("[EVAL] Fichiers manquants :")
            for m in miss: print(" -", m)

    if args.only in ("both", "train"):
        args.dst_train.mkdir(parents=True, exist_ok=True)
        n, miss = collect_train_csvs(args.src, args.dst_train)
        print(f"[TRAIN] Fichiers copiés : {n}")
        if miss:
            print("[TRAIN] Fichiers manquants :")
            for m in miss: print(" -", m)

if __name__ == "__main__":
    main()
