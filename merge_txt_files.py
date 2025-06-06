import os
import random

def load_all_texts(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip()]
                data.extend(lines)
    return data

def split_data(data, val_ratio=0.1, seed=0):
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    val_size = int(n * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    return train_data, val_data

def save_list_to_file(lines, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

# === Paramètres ===
input_folder = "train_env"
output_folder = "data"
val_test_folder = os.path.join(output_folder, "val_test")
train_output_file = os.path.join(output_folder, "train_shuffled.txt")

# === Création des dossiers ===
os.makedirs(val_test_folder, exist_ok=True)

# === Traitement ===
all_data = load_all_texts(input_folder)
train_data, val_data = split_data(all_data, val_ratio=0.1)

# === Sauvegarde ===
save_list_to_file(train_data, train_output_file)
save_list_to_file(val_data, os.path.join(val_test_folder, "val_ind.txt"))

print("Fichiers générés dans le dossier :", output_folder)
print(f" - {train_output_file}")
print(f" - {val_test_folder}/val.txt")
