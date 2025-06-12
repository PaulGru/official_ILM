import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def process_csv(file_path, model_type):
    df = pd.read_csv(file_path)

    # 1) Parser la run-id ("ilm_lr5e-05_seed0_steps7000")
    def parse_run_id(s):
        m = re.match(r'.*?_lr([0-9eE.+-]+)_seed(\d+)_steps(\d+)', s)
        if not m:
            return {'lr': None, 'seed': None, 'run_steps': None}
        return {
            'lr':         m.group(1),
            'seed':       int(m.group(2)),
            'run_steps':  int(m.group(3))
        }

    run_info = df['seed'].apply(parse_run_id).apply(pd.Series)
    df = pd.concat([df, run_info], axis=1)

    # 2) Parser l'étape d'évaluation depuis "model-XXXX"
    df['eval_steps'] = (
        df['model']
          .str.extract(r'model-(\d+)')
          .astype(int)
    )

    # Tag pour eLM vs iLM
    df['model_type'] = model_type
    return df

# --- Chargement des fichiers ---
df_elm = process_csv("csv/runs_elm/summary_data_eval_out_dist.csv", "eLM") #in
df_ilm = process_csv("csv/runs_ilm/summary_data_eval_out_dist.csv", "iLM")
df_all = pd.concat([df_elm, df_ilm], ignore_index=True)

# --- Calcul des moyennes et écarts-types sur eval_steps ---
grouped = (
    df_all
    .groupby(['model_type', 'eval_steps'])
    .agg(
        perplexity_mean=('perplexity', 'mean'),
        perplexity_std =('perplexity', 'std'),
        loss_mean      =('eval_loss', 'mean'),
        loss_std       =('eval_loss', 'std'),
    )
    .reset_index()
)

# --- Préparation du dossier de sortie ---
output_dir = "plots_output_eval_out_lr_5e-05" #in
os.makedirs(output_dir, exist_ok=True)

# --- Figure 1 : Perplexité ---
plt.figure(figsize=(10, 6))
for model_type, sub in grouped.groupby('model_type'):
    plt.errorbar(
        sub['eval_steps'],
        sub['perplexity_mean'],
        yerr=sub['perplexity_std'],
        label=model_type,
        marker='o',
        linestyle='-' if model_type == 'eLM' else '--'
    )

plt.xlabel("Étape d'évaluation (checkpoint)")
plt.ylabel("Perplexité moyenne")
plt.title("Perplexité (out-distribution) — eLM vs iLM") #in
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perplexity_out_comparaison.png")) #in
plt.show()

# --- Figure 2 : Loss d'évaluation ---
plt.figure(figsize=(10, 6))
for model_type, sub in grouped.groupby('model_type'):
    plt.errorbar(
        sub['eval_steps'],
        sub['loss_mean'],
        yerr=sub['loss_std'],
        label=model_type,
        marker='o',
        linestyle='-' if model_type == 'eLM' else '--'
    )

plt.xlabel("Étape d'évaluation (checkpoint)")
plt.ylabel("Perte d'évaluation moyenne")
plt.title("Evaluation Loss (out-distribution) — eLM vs iLM") #in
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_loss_ood_comparaison.png")) #ind
plt.show()
