import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main(sample_interval=100):
    # 1) Dossier racine où sont rangés les sous-dossiers d'entraînement
    base_dir = 'results_training'

    # 2) Pattern pour récupérer tous les train_env_losses.csv des modèles ilm
    pattern = os.path.join(base_dir, 'ilm_lr5e-05_seed*_steps*', 'train_env_losses.csv')
    file_paths = glob.glob(pattern)
    if not file_paths:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern : {pattern}")

    # 3) Charger et stocker toutes les DataFrames
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    # 4) Concaténer et moyenniser sur les seeds
    #    On suppose que tous les fichiers ont les mêmes colonnes et mêmes valeurs de 'step'
    df_all = pd.concat(dfs, ignore_index=True)
    df_mean = df_all.groupby('step').mean().reset_index()  # moyenne sur toutes les colonnes numériques

    df_sampled = df_mean[df_mean['step'] % sample_interval == 0]

    # 5) Préparer le dossier de sortie
    output_dir = 'results_plot_env_losses_ilm'
    os.makedirs(output_dir, exist_ok=True)

    # 6) Tracer les courbes
    fig, ax = plt.subplots(figsize=(8, 5))
    env_cols = [col for col in df_sampled.columns if col != 'step']
    for col in env_cols:
        ax.plot(df_sampled['step'], df_sampled[col], label=col)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train loss par environement')
    ax.set_title("Perte d'entraînement par environnement")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    # 7) Sauvegarder
    save_path = os.path.join(output_dir, 'train_env_losses_ilm.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Graphique enregistré : {save_path}")

if __name__ == '__main__':
    main()
