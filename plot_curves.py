# plot_training_and_eval.py
# -*- coding: utf-8 -*-

import os, glob, re, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ====== PARAMÈTRES =======
# =========================
LR_LIST = ["1e-05", "5e-05"]          # learning rates gardés distincts
MODELS  = ["elm", "ilm", "ilmg"]      # ordre d'affichage
SPLITS  = {
    "eval_in_dist": "In-Distribution",
    "eval_out_dist": "Out-of-Distribution"
}

# Répertoires d'entrée
EVAL_BASES   = {"elm": "results/saved_eval_elm", "ilm": "results/saved_eval_ilm", "ilmg": "results_1/saved_eval_ilmg",}
TRAIN_BASES  = {"elm": "results/saved_train_elm", "ilm": "results/saved_train_ilm", "ilmg": "results_1/saved_train_ilmg",}

# Répertoires de sortie
PLOTS_DIR = Path("plots_1")
EVAL_DIR  = PLOTS_DIR / "evaluation"
TRAIN_DIR = PLOTS_DIR / "training"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Lissage & sous-échantillonnage (pour les courbes d'entraînement)
SMOOTH_METHOD     = "ema"  # "ema" ou "ma"
EMA_ALPHA         = 0.05   # plus grand => plus lisse
MA_WINDOW         = 101    # taille fenêtre si "ma"
DOWNSAMPLE_EVERY  = 20     # garde 1 point sur N après lissage

# =========================
# ====== UTILITAIRES ======
# =========================
FLOAT = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
RE_LOSS = re.compile(rf"^\s*eval_loss\s*=\s*{FLOAT}\s*$")
RE_PPL  = re.compile(rf"^\s*perplexity\s*=\s*{FLOAT}\s*$")

# --- Alignement ELM pour comparabilité avec ILM (facultatif) ---
ALIGN_ELM_TO_ILM = True      # met False pour retrouver le comportement d'avant
ELM_ALIGN_STRIDE = 15        # taille du pas d'alignement
ELM_ALIGN_MODE   = "stride"  # "stride" (garder 1 point/15) ou "block_mean" (moyenne par blocs de 15)


def rebin_step_to_vals(step_to_vals: dict, stride: int, mode: str):
    """
    step_to_vals: {step -> [valeurs (une par seed)]}
    - 'stride'     : garde uniquement les steps multiples de 'stride'
    - 'block_mean' : regroupe les steps par blocs de taille 'stride' et concatène les valeurs
                     (ex: steps 1..15 -> bucket 15; 16..30 -> 30, etc.)
    Retourne un nouveau dict {new_step -> [valeurs...]}.
    """
    if not step_to_vals:
        return {}

    if mode == "stride":
        return {s: v for s, v in step_to_vals.items() if s % stride == 0}

    # block_mean
    bins = {}
    for s, vals in step_to_vals.items():
        # ceil au multiple de stride : 1..15 -> 15 ; 16..30 -> 30 ; etc.
        bucket = ((s + stride - 1) // stride) * stride
        bins.setdefault(bucket, []).extend(vals)
    return bins

def parse_eval_two_lines(path: Path):
    """Lit eval_results_mlm.txt (2 lignes: eval_loss, perplexity)."""
    loss = ppl = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_LOSS.match(line)
            if m: loss = float(m.group(1)); continue
            m = RE_PPL.match(line)
            if m: ppl  = float(m.group(1)); continue
    out = {}
    if loss is not None: out["eval_loss"] = loss
    if ppl  is not None: out["perplexity"] = ppl
    return out

def ema(y, alpha=0.1):
    if len(y) == 0: return np.array([], dtype=float)
    s = np.array(y, dtype=float).copy()
    for i in range(1, len(s)):
        s[i] = alpha * s[i] + (1 - alpha) * s[i - 1]
    return s

def moving_average_same(y, w=101):
    if w <= 1 or len(y) == 0: return np.array(y, dtype=float)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(np.array(y, dtype=float), kernel, mode="same")

def smooth_and_downsample(steps, means, stds):
    steps = np.asarray(steps)
    means = np.asarray(means)
    stds  = np.asarray(stds)
    if SMOOTH_METHOD == "ema":
        means_s = ema(means, EMA_ALPHA)
        stds_s  = ema(stds,  EMA_ALPHA)
    else:
        means_s = moving_average_same(means, MA_WINDOW)
        stds_s  = moving_average_same(stds,  MA_WINDOW)
    if DOWNSAMPLE_EVERY and DOWNSAMPLE_EVERY > 1:
        idx = slice(0, None, DOWNSAMPLE_EVERY)
        return steps[idx], means_s[idx], stds_s[idx]
    return steps, means_s, stds_s

# =========================================
# ====== PARTIE 1 : COURBES ÉVALUATION ====
# =========================================
def run_evaluation_plots():
    # Agrégats fins: (metric, model, lr, split, step) -> [values ...] (une par seed)
    values = defaultdict(list)
    found_any = False

    for model, base in EVAL_BASES.items():
        for lr in LR_LIST:
            pattern = f"{model}_lr{lr}_seed*_steps*"
            if model == "ilmg":
                pattern = f"{model}*_lr{lr}_seed*_steps*"
            run_dirs = sorted(glob.glob(os.path.join(base, pattern)))

            for run in run_dirs:
                for step_dir in sorted(glob.glob(os.path.join(run, "model-*"))):
                    try:
                        step = int(os.path.basename(step_dir).split("-")[-1])
                    except ValueError:
                        continue
                    for split in SPLITS.keys():
                        fpath = Path(step_dir) / split / "eval_results_mlm.txt"
                        if not fpath.is_file():
                            continue
                        metrics = parse_eval_two_lines(fpath)
                        if not metrics:
                            continue
                        found_any = True
                        for metric, val in metrics.items():
                            values[(metric, model, lr, split, step)].append(val)

    if not found_any:
        print("[WARN] Aucun fichier d'évaluation trouvé.")
        return

    # ---- (A) PLOTS par LR (comme avant) ----
    def plot_and_csv(metric_key: str):
        csv_path = EVAL_DIR / f"eval_agg_{metric_key}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["model", "lr", "split", "step", "mean", "std", "n"])

        for split, split_title in SPLITS.items():
            plt.figure()
            for model in MODELS:
                for lr in LR_LIST:
                    step_to_vals = {}
                    for (mk, m, l, s, step), vals in values.items():
                        if mk == metric_key and m == model and l == lr and s == split and len(vals) > 0:
                            step_to_vals[step] = vals
                    if not step_to_vals:
                        continue

                    steps = sorted(step_to_vals.keys())
                    means = np.array([np.mean(step_to_vals[st]) for st in steps], dtype=float)
                    stds  = np.array([np.std(step_to_vals[st])  for st in steps], dtype=float)
                    ns    = np.array([len(step_to_vals[st])     for st in steps], dtype=int)

                    label = f"{model.upper()} lr{lr}"
                    plt.plot(steps, means, label=label, linewidth=2)
                    plt.fill_between(steps, means - stds, means + stds, alpha=0.2)

                    with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
                        writer = csv.writer(fcsv)
                        for st, mu, sd, n in zip(steps, means, stds, ns):
                            writer.writerow([model, lr, split, st, float(mu), float(sd), int(n)])

            ylabel = "Perplexity" if "perplex" in metric_key else "Eval loss"
            plt.xlabel("Steps")
            plt.ylabel(ylabel)
            plt.title(f"{split_title} — moyenne sur seeds ({metric_key})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = EVAL_DIR / f"{'in_dist' if split=='eval_in_dist' else 'out_dist'}_{metric_key}.png"
            plt.savefig(out_png, dpi=200)
            print(f"[OK][Eval] Figure: {out_png}")

        print(f"[OK][Eval] CSV agrégé: {csv_path}")

    # ---- (B) NOUVEAU : PLOTS MOYENNÉS AUSSI SUR LR (2 courbes ELM/ILM) ----
    def plot_and_csv_lravg(metric_key: str):
        """
        Combine toutes les LR disponibles pour chaque (model, split, step).
        On moyenne donc sur 'seed' ET 'lr'.
        """
        csv_path = EVAL_DIR / f"eval_agg_{metric_key}_lravg.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["model", "split", "step", "mean", "std", "n"])

        for split, split_title in SPLITS.items():
            plt.figure()
            for model in MODELS:
                # regrouper step -> concat de toutes les valeurs sur toutes les LR
                step_to_vals = defaultdict(list)
                for lr in LR_LIST:
                    for (mk, m, l, s, step), vals in values.items():
                        if mk == metric_key and m == model and l == lr and s == split and len(vals) > 0:
                            step_to_vals[step].extend(vals)
                if not step_to_vals:
                    continue

                steps = sorted(step_to_vals.keys())
                means = np.array([np.mean(step_to_vals[st]) for st in steps], dtype=float)
                stds  = np.array([np.std(step_to_vals[st])  for st in steps], dtype=float)
                ns    = np.array([len(step_to_vals[st])     for st in steps], dtype=int)

                label = model.upper()
                plt.plot(steps, means, label=label, linewidth=2)
                plt.fill_between(steps, means - stds, means + stds, alpha=0.2)

                with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
                    writer = csv.writer(fcsv)
                    for st, mu, sd, n in zip(steps, means, stds, ns):
                        writer.writerow([model, split, st, float(mu), float(sd), int(n)])

            ylabel = "Perplexity" if "perplex" in metric_key else "Eval loss"
            plt.xlabel("Steps")
            plt.ylabel(ylabel)
            plt.title(f"{split_title} — moyenne sur seeds + LR ({metric_key})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = EVAL_DIR / f"{'in_dist' if split=='eval_in_dist' else 'out_dist'}_{metric_key}_lravg.png"
            plt.savefig(out_png, dpi=200)
            print(f"[OK][Eval] Figure (LR-avg): {out_png}")

        print(f"[OK][Eval] CSV agrégé (LR-avg): {csv_path}")

    # Figures
    plot_and_csv("perplexity")
    plot_and_csv("eval_loss")
    # Nouvelles figures à 2 courbes (ELM/ILM)
    plot_and_csv_lravg("perplexity")
    plot_and_csv_lravg("eval_loss")

# ==========================================
# ======= PARTIE 2 : COURBES TRAINING ======
# ==========================================
def run_training_plots():
    # total: (model, lr, step) -> [loss ...]
    total_vals = defaultdict(list)
    # ilm env: (lr, env, step) -> [loss ...]
    ilm_env_vals = defaultdict(list)
    ilm_env_names = set()

    found_any_total = False
    found_any_ilm_env = False

    for model, base in TRAIN_BASES.items():
        for lr in LR_LIST:
            run_dirs = sorted(glob.glob(os.path.join(base, f"{model}_lr{lr}_seed*_steps*")))
            for run in run_dirs:
                # total loss
                total_path = Path(run) / "train_total_loss.csv"
                if total_path.is_file():
                    try:
                        df = pd.read_csv(total_path)
                        if {"step","loss"}.issubset(df.columns):
                            found_any_total = True
                            for _, row in df.iterrows():
                                step = int(row["step"])
                                loss = float(row["loss"])
                                total_vals[(model, lr, step)].append(loss)
                    except Exception as e:
                        print(f"[WARN] Lecture {total_path}: {e}")

                # ILM par environnement
                if model == "ilm":
                    env_path = Path(run) / "train_env_losses.csv"
                    if env_path.is_file():
                        try:
                            df_env = pd.read_csv(env_path)
                            if "step" in df_env.columns:
                                found_any_ilm_env = True
                                env_cols = [c for c in df_env.columns if c != "step"]
                                for env in env_cols:
                                    ilm_env_names.add(env)
                                    for _, row in df_env.iterrows():
                                        step = int(row["step"])
                                        val = float(row[env])
                                        ilm_env_vals[(lr, env, step)].append(val)
                        except Exception as e:
                            print(f"[WARN] Lecture {env_path}: {e}")

    if not found_any_total:
        print("[WARN] Aucun train_total_loss.csv trouvé.")
        return

    # ---- TOTAL LOSS (ELM & ILM) ----
    csv_total = TRAIN_DIR / "train_total_loss_agg.csv"
    with csv_total.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["model", "lr", "step", "mean", "std", "n"])

    plt.figure()
    for model in MODELS:
        for lr in LR_LIST:
            step_to_vals = {step: vals for (m, l, step), vals in total_vals.items()
                            if m == model and l == lr and len(vals) > 0}
            if not step_to_vals:
                continue
            
            if ALIGN_ELM_TO_ILM and model == "elm":
                step_to_vals = rebin_step_to_vals(step_to_vals, ELM_ALIGN_STRIDE, ELM_ALIGN_MODE)

            steps = sorted(step_to_vals.keys())
            means = np.array([np.mean(step_to_vals[s]) for s in steps], dtype=float)
            stds  = np.array([np.std(step_to_vals[s])  for s in steps], dtype=float)
            ns    = np.array([len(step_to_vals[s])     for s in steps], dtype=int)

            # CSV (valeurs brutes)
            with csv_total.open("a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                for s, mu, sd, n in zip(steps, means, stds, ns):
                    writer.writerow([model, lr, s, float(mu), float(sd), int(n)])

            # Lissage + downsample pour affichage
            steps_s, means_s, stds_s = smooth_and_downsample(steps, means, stds)

            label = f"{model.upper()} lr{lr}"
            plt.plot(steps_s, means_s, label=label, linewidth=2)
            plt.fill_between(steps_s, means_s - stds_s, means_s + stds_s, alpha=0.15)

    plt.xlabel("Steps")
    plt.ylabel("Training loss")
    plt.title("Training — total loss (moyenne sur seeds, lissé)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png_total = TRAIN_DIR / "train_total_loss_smoothed.png"
    plt.savefig(out_png_total, dpi=200)
    print(f"[OK][Train] Figure: {out_png_total}")
    print(f"[OK][Train] CSV (brut): {csv_total}")

    # ---- ILM par environnement (lissé) ----
    if found_any_ilm_env:
        for lr in LR_LIST:
            env_to_steps = {}
            for env in sorted(ilm_env_names):
                step_to_vals = {step: vals for (l, e, step), vals in ilm_env_vals.items()
                                if l == lr and e == env and len(vals) > 0}
                if step_to_vals:
                    env_to_steps[env] = step_to_vals
            if not env_to_steps:
                continue

            # CSV agrégé
            csv_env = TRAIN_DIR / f"train_ilm_env_losses_lr{lr}_agg.csv"
            with csv_env.open("w", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["lr", "environment", "step", "mean", "std", "n"])

            plt.figure()
            for env, step_to_vals in env_to_steps.items():
                steps = sorted(step_to_vals.keys())
                means = np.array([np.mean(step_to_vals[s]) for s in steps], dtype=float)
                stds  = np.array([np.std(step_to_vals[s])  for s in steps], dtype=float)
                # lissage + downsample
                steps_s, means_s, _ = smooth_and_downsample(steps, means, stds)
                plt.plot(steps_s, means_s, label=env)

                with csv_env.open("a", newline="", encoding="utf-8") as fcsv:
                    writer = csv.writer(fcsv)
                    for s, mu, sd in zip(steps, means, stds):
                        writer.writerow([lr, env, int(s), float(mu), float(sd), int(len(step_to_vals[s]))])

            plt.xlabel("Steps")
            plt.ylabel("Training env loss")
            plt.title(f"Training — ILM par environnement (lr {lr}) — moyenne sur seeds, lissé")
            plt.legend(fontsize=8, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png_env = TRAIN_DIR / f"train_ilm_env_losses_lr{lr}_smoothed.png"
            plt.savefig(out_png_env, dpi=200)
            print(f"[OK][Train] Figure: {out_png_env}")
            print(f"[OK][Train] CSV:    {csv_env}")
    else:
        print("[INFO][Train] Aucun train_env_losses.csv ILM trouvé — pas de figures 'par environnement'.")

# =========================
# ========= MAIN ==========
# =========================
if __name__ == "__main__":
    run_evaluation_plots()
    run_training_plots()
    print("[DONE] Courbes d'évaluation (avec et sans moyenne sur LR) et d'entraînement générées dans 'plots/evaluation' et 'plots/training'.")
