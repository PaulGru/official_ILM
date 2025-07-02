import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Créer le dossier pour les résultats
    output_dir = 'results_plot_eval'
    os.makedirs(output_dir, exist_ok=True)

    # Liste des fichiers CSV (adapter les chemins si besoin)
    files = [
        'results/all_eval_results_elm.csv',
        'results/all_eval_results_ilm.csv',
        'results/all_eval_results_ilmg.csv'
    ]

    # Charger et concaténer les données
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Moyenne de perplexité par modèle, eval_type et step
    grouped = (
        df
        .groupby(['model', 'eval_type', 'step'], as_index=False)
        ['perplexity']
        .mean()
    )

    # Générer et sauvegarder les plots
    for eval_type in ['eval_in_dist', 'eval_out_dist']:
        subset = grouped[grouped['eval_type'] == eval_type]
        fig, ax = plt.subplots()
        for model in subset['model'].unique():
            mdf = subset[subset['model'] == model]
            ax.plot(mdf['step'], mdf['perplexity'], label=model)
        ax.set_xlabel('Step')
        ax.set_ylabel('Perplexity')
        ax.set_title(f'Perplexity vs Step ({eval_type})')
        ax.legend()
        filepath = os.path.join(output_dir, f'{eval_type}.png')
        plt.savefig(filepath)
        plt.close(fig)
        print(f'Saved plot for {eval_type} → {filepath}')

if __name__ == '__main__':
    main()
