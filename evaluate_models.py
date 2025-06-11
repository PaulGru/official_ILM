import os
import json
import math
import subprocess
import shutil
import pandas as pd

base_dir = "runs_elm"
# 1) On déclare nos deux jeux de test
test_sets = {
    "in_dist":  "data/val_test/val_ind.txt",
    "out_dist": "data/val_test/val_ood.txt"
}

# 2) On prépare un dictionnaire pour collecter les résultats
all_records = { mode: [] for mode in test_sets }

common_args = [
    "--tokenizer_name", "distilbert-base-uncased",
    "--do_eval",
    "--max_seq_length", "128",
    "--per_device_eval_batch_size", "4"
]

# 3) On boucle d'abord sur chaque mode (in/out)
for mode, test_file in test_sets.items():
    print(f"\n▶ Lancement des évaluations [{mode}] avec `{test_file}`\n")
    for seed_dir in sorted(os.listdir(base_dir)):
        seed_path = os.path.join(base_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        for model_folder in sorted(os.listdir(seed_path)):
            if not model_folder.startswith("model-"):
                continue

            model_path = os.path.join(seed_path, model_folder)
            print(f"[{mode}] Seed `{seed_dir}` – Modèle `{model_folder}`")

            eval_out = os.path.join(model_path, f"eval_{mode}")
            os.makedirs(eval_out, exist_ok=True)

            cmd = [
                "python", "run_invariant_mlm.py",
                "--model_name_or_path", model_path,
                "--output_dir",            eval_out,
                "--validation_file",       test_file
            ] + common_args

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Échec pour {model_folder} : {e}")
                continue

            default_result = os.path.join(eval_out, "eval_results.json")
            if not os.path.exists(default_result):
                print(f"Pas de eval_results.json pour {model_folder}")
                continue

            # copie d’archive (optionnelle)
            archived = os.path.join(model_path, f"eval_{mode}_results.json")
            shutil.copy(default_result, archived)

            metrics = json.load(open(default_result))
            loss   = metrics.get("eval_loss")
            perp   = math.exp(loss) if loss is not None else None

            all_records[mode].append({
                "seed":       seed_dir,
                "model":      model_folder,
                "eval_loss":  loss,
                "perplexity": perp
            })

# 4) On écrit un CSV par mode
for mode, records in all_records.items():
    df     = pd.DataFrame(records)
    out_csv = os.path.join(base_dir, f"summary_data_eval_{mode}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Résultats {mode} → `{out_csv}`")
