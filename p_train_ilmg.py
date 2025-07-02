import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_ilmg_training(base_dir='results_training', output_dir='results_training_ilmg', sample_interval=100):
    """
    Pour le modèle ilmg :
    - Moyenne la loss totale par step sur toutes les seeds disponibles et trace la courbe.
    - Moyenne les losses par environnement par step et trace les courbes.
    Enregistre les deux PNG dans output_dir.
    """
    model = 'ilmg'
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Train total loss ---
    # Récupère tous les train_total_loss.csv pour ilmg
    total_pattern = os.path.join(
        base_dir, f'{model}_lr5e-05_seed*_steps*', 'train_total_loss.csv'
    )
    total_files = glob.glob(total_pattern)
    if not total_files:
        raise FileNotFoundError(f"Aucun train_total_loss.csv trouvé ({total_pattern})")

    # Charge et concatène
    dfs_total = [pd.read_csv(fp) for fp in total_files]
    df_total_all = pd.concat(dfs_total, ignore_index=True)

    # Moyenne sur les seeds
    df_total_mean = df_total_all.groupby('step', as_index=False)['loss'].mean()

    df_total_sampled = df_total_mean[df_total_mean['step'] % sample_interval == 0]

    # Trace et enregistre
    plt.figure(figsize=(8,5))
    plt.plot(df_total_sampled['step'], df_total_sampled['loss'], label='ilmg')
    plt.xlabel('Step')
    plt.ylabel('Train total loss')
    plt.title('ilmg – perte totale d’entraînement')
    plt.legend()
    plt.tight_layout()
    total_out = os.path.join(output_dir, 'train_total_loss_ilmg.png')
    plt.savefig(total_out)
    plt.close()

    print(f"Enregistré : {total_out}")

    # --- 2) Train env losses ---
    # Récupère tous les train_env_losses.csv pour ilmg
    env_pattern = os.path.join(
        base_dir, f'{model}_lr5e-05_seed*_steps*', 'train_env_losses.csv'
    )
    env_files = glob.glob(env_pattern)
    if not env_files:
        raise FileNotFoundError(f"Aucun train_env_losses.csv trouvé ({env_pattern})")

    # Charge et concatène
    dfs_env = [pd.read_csv(fp) for fp in env_files]
    df_env_all = pd.concat(dfs_env, ignore_index=True)

    # Moyenne par step
    df_env_mean = df_env_all.groupby('step', as_index=False).mean()

    df_env_sampled = df_env_mean[df_env_mean['step'] % sample_interval == 0]

    # Trace une courbe par environnement (colonnes autres que 'step')
    plt.figure(figsize=(8,5))
    env_cols = [c for c in df_env_sampled.columns if c != 'step']
    for col in env_cols:
        plt.plot(df_env_sampled['step'], df_env_sampled[col], label=col)
    plt.xlabel('Step')
    plt.ylabel('Train env loss')
    plt.title('ilmg – perte d’entraînement par environnement')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    env_out = os.path.join(output_dir, 'train_env_losses_ilmg.png')
    plt.savefig(env_out)
    plt.close()

    print(f"Enregistré : {env_out}")

if __name__ == '__main__':
    plot_ilmg_training()
