import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = 'results_training'
    models   = ['elm', 'ilm']
    sample_interval = 100  # ne garder qu’un point tous les 100 steps

    model_means = {}
    for model in models:
        pattern = os.path.join(base_dir,
                               f'{model}_lr5e-05_seed*_steps*',
                               'train_total_loss.csv')
        file_paths = glob.glob(pattern)
        if not file_paths:
            print(f"Aucun fichier trouvé pour {model}")
            continue

        # Concat & moyenne par step
        dfs = [pd.read_csv(p) for p in file_paths]
        df_all = pd.concat(dfs, ignore_index=True)
        df_mean = df_all.groupby('step', as_index=False)['loss'].mean()

        # Échantillonnage tous les `sample_interval` steps
        df_mean = df_mean[df_mean['step'] % sample_interval == 0]
        model_means[model] = df_mean

    # Préparation du dossier de sortie
    output_dir = 'results_plot_total_loss'
    os.makedirs(output_dir, exist_ok=True)

    # Tracé
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, df_mean in model_means.items():
        ax.plot(df_mean['step'], df_mean['loss'], label=model)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train total loss (moyenne sur seeds)')
    ax.set_title(f"Perte totale d'entraînement")
    ax.legend()
    plt.tight_layout()

    # Sauvegarde
    save_path = os.path.join(output_dir, 'train_total_loss_models_sampled.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Graphique enregistré → {save_path}")

if __name__ == '__main__':
    main()
