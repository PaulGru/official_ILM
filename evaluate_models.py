import os
import json
import math
import subprocess
import shutil
import pandas as pd


base_dir = "runs_elm" # ilm
custom_test_file = "data/val_test/val_ood.txt" # test.txt
output_csv = os.path.join(base_dir, "summary_data_eval_ood.csv") # test

common_args = [
    "--validation_file", custom_test_file,
    "--tokenizer_name", "distilbert-base-uncased",
    "--do_eval",
    "--max_seq_length", "128",
    "--per_device_eval_batch_size", "4",
    "--line_by_line", "False"
]

records = []

for model_name in sorted(os.listdir(base_dir)):
    model_path = os.path.join(base_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    print(f"[EVAL] Évaluation du modèle : {model_name}")

    cmd = [
        "python", "run_invariant_mlm.py",
        "--model_name_or_path", model_path,
        "--output_dir", model_path
    ] + common_args

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Échec de l'évaluation pour {model_name} : {e}")
        continue

    default_result = os.path.join(model_path, "eval_results.json")
    custom_result = os.path.join(model_path, "data_ood_results.json") # data_test_results

    if os.path.exists(default_result):
        shutil.copy(default_result, custom_result)
    else:
        print(f"Fichier eval_results.json manquant pour {model_name}")
        continue

    with open(custom_result) as f:
        metrics = json.load(f)

    loss = metrics.get("eval_loss", None)
    perplexity = None
    try:
        if loss is not None:
            perplexity = math.exp(loss)
    except Exception:
        pass

    records.append({
        "model": model_name,
        "eval_loss": loss,
        "perplexity": perplexity
    })

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Résultats sauvegardés dans : {output_csv}")
