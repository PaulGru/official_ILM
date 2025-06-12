import os
import json
import math
import subprocess
import time
from itertools import product
import pandas as pd

# ------------------ CONFIG ------------------
learning_rates = [5e-5] # 1e-6, seed pourri, 1e-5 pas mal mais moins bonne que 5e-5
seeds = [123]
nb_steps = [5000]

base_dirs = {
    "ilm": "runs_ilm",
    "elm": "runs_elm"
}

# ------------------ TRAINING ------------------
def launch_training(model_key):
    base_dir = base_dirs[model_key]

    if model_key == "ilm":
        train_file = "data/train_env"
    elif model_key == "elm":
        train_file = "data"

    os.makedirs(base_dir, exist_ok=True)
    for lr, seed, step in product(learning_rates, seeds, nb_steps):
        exp_name = f"{model_key}_lr{lr}_seed{seed}_steps{step}"
        out_dir = os.path.join(base_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "run_invariant_mlm.py",
            "--model_name_or_path", "distilbert-base-uncased",
            "--do_train",
            "--train_file", train_file,
            "--validation_file", "data/val_test/val_ind.txt",
            "--output_dir", out_dir,
            "--overwrite_output_dir",
            "--max_seq_length", "128",
            "--preprocessing_num_workers", "16",
            "--per_device_train_batch_size", "16",
            "--gradient_accumulation_steps", "3",
            "--learning_rate", str(lr),
            "--save_total_limit", "60",
            "--logging_steps", "1000",
            "--nb_steps_model_saving", "1000",
            "--nb_steps", str(step),
            "--fp16",
            "--seed", str(seed)
        ]

        print(f"[TRAIN] Launching: {exp_name}")
        subprocess.run(cmd)
        print(f"[TRAIN] Finished: {exp_name}\n")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    t0 = time.time()
    print("=== Lancement des entraînements ELM ===")
    launch_training("elm")
    print("=== Lancement des entraînements ILM ===")
    launch_training("ilm")
    print(f"[DONE] Tous les entraînements terminés en {round(time.time() - t0, 2)}s")
