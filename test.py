import os
import random
from pathlib import Path

random.seed(0)  # pour reproductibilité

RAW_DATA_DIR = "train_env"       # dossier avec les fichiers .txt par environnement
VAL_DIR = "data/val_test"
TRAIN_ENV_DIR = "data/train_env"
ERM_FILE = "data/train_erm.txt"

os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TRAIN_ENV_DIR, exist_ok=True)

val_lines = []
erm_lines = []

for file_name in os.listdir(RAW_DATA_DIR):
    if not file_name.endswith(".txt"):
        continue

    env_path = os.path.join(RAW_DATA_DIR, file_name)
    with open(env_path, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)
    n = len(lines)
    val_size = int(n * 0.10)

    val_split = lines[:val_size]
    train_split = lines[val_size:]

    # 1. Ajoute au val_ind
    val_lines.extend(val_split)

    # 2. Écrit 90% dans data/train_env/<env>.txt
    out_env_path = os.path.join(TRAIN_ENV_DIR, file_name)
    with open(out_env_path, "w") as f_out:
        f_out.writelines(train_split)

    # 3. Ajoute au ERM
    erm_lines.extend(train_split)

# Écrit val_ind.txt
with open(os.path.join(VAL_DIR, "val_ind.txt"), "w") as f_val:
    f_val.writelines(val_lines)

# Shuffle global ERM
random.shuffle(erm_lines)
with open(ERM_FILE, "w") as f_erm:
    f_erm.writelines(erm_lines)

print("✅ Données préparées :")
print(f"- Validation : {len(val_lines)} lignes dans {VAL_DIR}/val_ind.txt")
print(f"- Environnements : {len(os.listdir(TRAIN_ENV_DIR))} fichiers dans {TRAIN_ENV_DIR}/")
print(f"- ERM total : {len(erm_lines)} lignes dans {ERM_FILE}")
