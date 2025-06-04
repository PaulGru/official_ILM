import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def process_csv(file_path, model_type):
    df = pd.read_csv(file_path)

    def parse_model_name(model_name):
        lr_match = re.search(r'lr([0-9a-zA-Z.-]+)', model_name)
        seed_match = re.search(r'seed(\d+)', model_name)
        steps_match = re.search(r'steps(\d+)', model_name)
        return {
            'lr': lr_match.group(1) if lr_match else None,
            'seed': int(seed_match.group(1)) if seed_match else None,
            'steps': int(steps_match.group(1)) if steps_match else None
        }

    parsed_data = df['model'].apply(parse_model_name).apply(pd.Series)
    df_parsed = pd.concat([df, parsed_data], axis=1)
    df_parsed['model_type'] = model_type
    return df_parsed

file_elm = "runs_elm/summary_data_test.csv" #summary_data_eval_ood.csv, summary_results.csv
file_ilm = "runs_ilm/summary_data_test.csv"  

df_elm = process_csv(file_elm, "eLM")
df_ilm = process_csv(file_ilm, "iLM")
df_all = pd.concat([df_elm, df_ilm], ignore_index=True)

grouped = df_all.groupby(['model_type', 'lr', 'steps']).agg(
    perplexity_mean=('perplexity', 'mean'),
    loss_mean=('eval_loss', 'mean')
).reset_index()

output_dir = "plots_output_test" # ood, validation
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
for model_type, style in zip(['eLM', 'iLM'], ['solid', 'dashed']):
    for lr in grouped['lr'].unique():
        sub_df = grouped[(grouped['model_type'] == model_type) & (grouped['lr'] == lr)].sort_values('steps')
        plt.plot(
            sub_df['steps'],
            sub_df['perplexity_mean'],
            label=f'{model_type} lr={lr}',
            linestyle=style,
            marker='o'
        )

plt.xlabel("Training Steps")
plt.ylabel("Perplexité moyenne")
plt.title("Perplexity sur ind selon eLM/iLM (les résultats sont moyennés par seed (0,1,2,3))") # ood
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "perplexity_comparison.png"))
plt.show()

plt.figure(figsize=(10, 6))
for model_type, style in zip(['eLM', 'iLM'], ['solid', 'dashed']):
    for lr in grouped['lr'].unique():
        sub_df = grouped[(grouped['model_type'] == model_type) & (grouped['lr'] == lr)].sort_values('steps')
        plt.errorbar(
            sub_df['steps'],
            sub_df['loss_mean'],
            label=f'{model_type} lr={lr}',
            linestyle=style,
            marker='o'
        )

plt.xlabel("Training Steps")
plt.ylabel("Perte d'évaluation")
plt.title("Evaluation Loss sur ins selon eLM/iLM (les résultats sont moyennés par seed (0,1,2,3))") # ood
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_loss_comparison.png"))
plt.show()
