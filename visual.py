'''
1. 模型在不同数据集上的 NDCG/Recall/Precision 曲线
思路：
运行模型时分别用不同数据集(如 ml-1m、amazon-book、yelp2018)。
每个 run 都会生成一个 log 文件，里面有每 5 个 epoch 的 valid/test NDCG/Recall/Precision。
读取这些 log 文件，画出曲线对比。
'''
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from sklearn.manifold import TSNE

# --- Config ---
datasets = ["ml-1m", "amazon-book", "yelp2018", "lastfm", "gowalla"]
models = ["lgn", "mf"]
log_dir = "code/logs/"
emb_dir = "code/embs/"
output_dir = "output"
# For lgn, log/emb file has _nl2, for mf it does not
lgn_suffix = "_lgn_dim64_lr0.001_dec0.0001_alpha0.6_beta-0.1_gamma0.2_nl2"
mf_suffix = "_mf_dim64_lr0.001_dec0.0001_alpha0.6_beta-0.1_gamma0.2"
metric_idx = {"ndcg": 1, "recall": 3, "precision": 5}

# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

# --- 1. Curves: NDCG@20 for both models on each dataset ---
for dataset in datasets:
    plt.figure(figsize=(10,6))
    for model in models:
        if model == "lgn":
            log_file = f"{log_dir}/{dataset}_seed2020{lgn_suffix}.txt"
        else:
            log_file = f"{log_dir}/{dataset}_seed2020{mf_suffix}.txt"
        epochs, ndcg = [], []
        try:
            with open(log_file, "r") as f:
                for idx, line in enumerate(f):
                    if line.startswith("valid"):
                        parts = line.strip().split()
                        epochs.append(idx*5)
                        ndcg.append(float(parts[metric_idx["ndcg"]]))
            plt.plot(epochs, ndcg, label=model, marker='o', markersize=3)
        except FileNotFoundError:
            print(f"File {log_file} not found")
            continue
    plt.xlabel("Epoch")
    plt.ylabel("NDCG@20")
    plt.title(f"NDCG@20 on {dataset} (lgn vs mf)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ndcg_comparison_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- 2. Bar charts: Best metrics for both models across datasets ---
results = {model: {"ndcg": [], "recall": [], "precision": []} for model in models}
summary_table = []
for dataset in datasets:
    row = [dataset]
    for model in models:
        if model == "lgn":
            log_file = f"{log_dir}/{dataset}_seed2020{lgn_suffix}.txt"
        else:
            log_file = f"{log_dir}/{dataset}_seed2020{mf_suffix}.txt"
        best_ndcg, best_recall, best_precision = 0, 0, 0
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.startswith("test"):
                        parts = line.strip().split()
                        ndcg = float(parts[1])
                        recall = float(parts[3])
                        precision = float(parts[5])
                        if ndcg > best_ndcg:
                            best_ndcg = ndcg
                            best_recall = recall
                            best_precision = precision
        except FileNotFoundError:
            print(f"File {log_file} not found")
        results[model]["ndcg"].append(best_ndcg)
        results[model]["recall"].append(best_recall)
        results[model]["precision"].append(best_precision)
        row.extend([best_ndcg, best_recall, best_precision])
    summary_table.append(row)

# Plot bar charts for each metric
metrics = ["ndcg", "recall", "precision"]
for metric in metrics:
    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width/2, results["lgn"][metric], width, label='lgn')
    rects2 = ax.bar(x + width/2, results["mf"][metric], width, label='mf')
    ax.set_ylabel(f"Best {metric.upper()}@20")
    ax.set_title(f"Best {metric.upper()}@20 Comparison Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_bar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- 3. Print summary table ---
print("\nSummary Table (dataset, lgn_ndcg, lgn_recall, lgn_precision, mf_ndcg, mf_recall, mf_precision):")
for row in summary_table:
    print("\t".join([str(x) for x in row]))

# --- 4. Embedding visualization for both models ---
for dataset in datasets:
    for model in models:
        if model == "lgn":
            emb_file = f"{emb_dir}{dataset}_seed2020{lgn_suffix}.pkl"
        else:
            emb_file = f"{emb_dir}{dataset}_seed2020{mf_suffix}.pkl"
        if not os.path.exists(emb_file):
            print(f"Embedding file {emb_file} not found")
            continue
        try:
            with open(emb_file, "rb") as f:
                embeddings = pickle.load(f)
                if len(embeddings) >= 2:
                    all_users, all_items = embeddings[:2]
                else:
                    print(f"Embedding file {emb_file} has insufficient data")
                    continue
            user_emb = all_users[:1000]
            item_emb = all_items[:1000]
            X = np.vstack([user_emb, item_emb])
            labels = np.array([0]*len(user_emb) + [1]*len(item_emb))
            print(f"Running t-SNE for {dataset} ({model})...")
            tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
            X_2d = tsne.fit_transform(X)
            plt.figure(figsize=(10,8))
            plt.scatter(X_2d[labels==0,0], X_2d[labels==0,1], c='blue', label='User', alpha=0.6, s=20)
            plt.scatter(X_2d[labels==1,0], X_2d[labels==1,1], c='red', label='Item', alpha=0.6, s=20)
            plt.legend()
            plt.title(f"User/Item Embedding Visualization (t-SNE) - {dataset} ({model})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'embedding_{dataset}_{model}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error processing {emb_file}: {e}")
            continue