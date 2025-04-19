import torch
import os
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib
matplotlib.use('Agg') 

# load clip and dinov2 models
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    print("Loading DINOv2 model...")
    dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)
    dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    
    return device, clip_model, clip_processor, dinov2_model, dinov2_processor

def get_embeddings(image_path, device, clip_model, clip_processor, dinov2_model, dinov2_processor):
    image = Image.open(image_path).convert('RGB')
    
    clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_embedding = clip_model.get_image_features(**clip_inputs)
    
    dinov2_inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        dinov2_output = dinov2_model(**dinov2_inputs)
        dinov2_embedding = dinov2_output.last_hidden_state[:, 0]
    
    return clip_embedding.cpu().numpy(), dinov2_embedding.cpu().numpy()

def evaluate_clustering(embeddings, labels, method_name):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        print(f"Not enough clusters for {method_name} (found {n_clusters})")
        return {}
    
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_id[label] for label in labels])
    
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    except Exception as e:
        print(f"Error calculating silhouette score for {method_name}: {e}")
        metrics['silhouette'] = float('nan')
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
    except Exception as e:
        print(f"Error calculating Calinski-Harabasz score for {method_name}: {e}")
        metrics['calinski_harabasz'] = float('nan')
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
    except Exception as e:
        print(f"Error calculating Davies-Bouldin score for {method_name}: {e}")
        metrics['davies_bouldin'] = float('nan')
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        metrics['adjusted_rand_index'] = adjusted_rand_score(int_labels, kmeans_labels)
    except Exception as e:
        print(f"Error calculating Adjusted Rand Index for {method_name}: {e}")
        metrics['adjusted_rand_index'] = float('nan')
        
    try:
        intra_dists = []
        for label in unique_labels:
            cluster_points = embeddings[np.array(labels) == label]
            if len(cluster_points) > 1:
                centroid = np.mean(cluster_points, axis=0)
                dists = np.linalg.norm(cluster_points - centroid, axis=1)
                intra_dists.append(np.mean(dists))
        
        if intra_dists:
            metrics['intra_cluster_distance'] = np.mean(intra_dists)
        else:
            metrics['intra_cluster_distance'] = float('nan')
    except Exception as e:
        print(f"Error calculating intra-cluster distance for {method_name}: {e}")
        metrics['intra_cluster_distance'] = float('nan')
        
    try:
        centroids = []
        for label in unique_labels:
            cluster_points = embeddings[np.array(labels) == label]
            centroids.append(np.mean(cluster_points, axis=0))
        
        inter_dists = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        
        if inter_dists:
            metrics['inter_cluster_distance'] = np.mean(inter_dists)
        else:
            metrics['inter_cluster_distance'] = float('nan')
    except Exception as e:
        print(f"Error calculating inter-cluster distance for {method_name}: {e}")
        metrics['inter_cluster_distance'] = float('nan')
    
    return metrics

def create_individual_cluster_visualization(clip_2d, dinov2_2d, labels, clip_embeddings, dinov2_embeddings):
    unique_labels = np.unique(labels)
    
    selected_labels = unique_labels[:min(4, len(unique_labels))]
    print(f"Creating individual cluster visualizations for: {', '.join(selected_labels)}")
    
    fig, axes = plt.subplots(len(selected_labels), 2, figsize=(18, 5*len(selected_labels)))
    
    if len(selected_labels) == 1:
        axes = axes.reshape(1, 2)
    
    clip_avg_distances = {}
    dinov2_avg_distances = {}
    
    for i, label in enumerate(selected_labels):
        label_indices = np.where(np.array(labels) == label)[0]
        
        clip_cluster = clip_2d[label_indices]
        dinov2_cluster = dinov2_2d[label_indices]
        
        clip_centroid = np.mean(clip_cluster, axis=0)
        dinov2_centroid = np.mean(dinov2_cluster, axis=0)
        
        clip_distances = np.linalg.norm(clip_cluster - clip_centroid, axis=1)
        dinov2_distances = np.linalg.norm(dinov2_cluster - dinov2_centroid, axis=1)
        
        clip_avg_distances[label] = np.mean(clip_distances)
        dinov2_avg_distances[label] = np.mean(dinov2_distances)
        
        clip_cluster_orig = clip_embeddings[label_indices]
        dinov2_cluster_orig = dinov2_embeddings[label_indices]
        
        clip_centroid_orig = np.mean(clip_cluster_orig, axis=0)
        dinov2_centroid_orig = np.mean(dinov2_cluster_orig, axis=0)
        
        clip_distances_orig = np.linalg.norm(clip_cluster_orig - clip_centroid_orig, axis=1)
        dinov2_distances_orig = np.linalg.norm(dinov2_cluster_orig - dinov2_centroid_orig, axis=1)
        
        axes[i, 0].scatter(clip_cluster[:, 0], clip_cluster[:, 1], alpha=0.7, s=100)
        axes[i, 0].scatter(clip_centroid[0], clip_centroid[1], color='red', marker='X', s=200, label='Centroid')
        
        sample_indices = np.random.choice(len(clip_cluster), min(10, len(clip_cluster)), replace=False)
        for idx in sample_indices:
            axes[i, 0].plot([clip_cluster[idx, 0], clip_centroid[0]], 
                          [clip_cluster[idx, 1], clip_centroid[1]], 
                          'k-', alpha=0.2)
            
        axes[i, 0].set_title(f'CLIP - {label} Cluster\nAvg Distance in t-SNE: {clip_avg_distances[label]:.2f}\nAvg Distance in Original Space: {np.mean(clip_distances_orig):.2f}')
        axes[i, 0].legend()
        
        axes[i, 1].scatter(dinov2_cluster[:, 0], dinov2_cluster[:, 1], alpha=0.7, s=100)
        axes[i, 1].scatter(dinov2_centroid[0], dinov2_centroid[1], color='red', marker='X', s=200, label='Centroid')
        
        for idx in sample_indices:
            axes[i, 1].plot([dinov2_cluster[idx, 0], dinov2_centroid[0]], 
                          [dinov2_cluster[idx, 1], dinov2_centroid[1]], 
                          'k-', alpha=0.2)
        
        axes[i, 1].set_title(f'DINOv2 - {label} Cluster\nAvg Distance in t-SNE: {dinov2_avg_distances[label]:.2f}\nAvg Distance in Original Space: {np.mean(dinov2_distances_orig):.2f}')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('individual_cluster_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved individual cluster visualization to individual_cluster_comparison.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(selected_labels))
    
    clip_distances = [clip_avg_distances[label] for label in selected_labels]
    dinov2_distances = [dinov2_avg_distances[label] for label in selected_labels]
    
    plt.bar(x - width/2, clip_distances, width, label='CLIP')
    plt.bar(x + width/2, dinov2_distances, width, label='DINOv2')
    
    plt.xlabel('Transformation Type')
    plt.ylabel('Average Distance to Centroid (t-SNE space)')
    plt.title('Intra-Cluster Point Distribution Comparison')
    plt.xticks(x, selected_labels, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('intra_cluster_distance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved intra-cluster distance chart to intra_cluster_distance_comparison.png")
    plt.close()
    
    return clip_avg_distances, dinov2_avg_distances

def create_per_transformation_clustering_chart(clip_cluster_distances, dinov2_cluster_distances):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(clip_cluster_distances.keys())
    clip_vals = [clip_cluster_distances[label] for label in labels]
    dino_vals = [dinov2_cluster_distances[label] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, clip_vals, width, label='CLIP')
    bars2 = plt.bar(x + width/2, dino_vals, width, label='DINOv2')

    plt.xlabel("Transformation Type")
    plt.ylabel("Avg Distance to Centroid (Original Embedding Space)")
    plt.title("Per-Transformation Clustering Performance")
    plt.xticks(x, labels, rotation=45)
    plt.legend()

    worst_clip_idx = np.argmax(clip_vals)
    worst_dino_idx = np.argmax(dino_vals)

    plt.text(x[worst_clip_idx] - width/2, clip_vals[worst_clip_idx] + 1, "Worst CLIP", ha='center', fontsize=10, color='red')
    plt.text(x[worst_dino_idx] + width/2, dino_vals[worst_dino_idx] + 1, "Worst DINOv2", ha='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('per_transformation_clustering.png', dpi=300)
    print("Saved per-transformation clustering comparison to per_transformation_clustering.png")
    plt.close()

def plot_colored_cluster_spread(clip_2d, dinov2_2d, labels, title="Colored Cluster Spread"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df_clip = pd.DataFrame(clip_2d, columns=["x", "y"])
    df_clip["transformation"] = labels

    df_dino = pd.DataFrame(dinov2_2d, columns=["x", "y"])
    df_dino["transformation"] = labels

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_clip, x="x", y="y", hue="transformation", palette="Set2", s=80, alpha=0.8)
    plt.title("CLIP: Colored Cluster Spread", fontsize=16)
    plt.xlabel("")
    plt.ylabel("")

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_dino, x="x", y="y", hue="transformation", palette="Set2", s=80, alpha=0.8)
    plt.title("DINOv2: Colored Cluster Spread", fontsize=16)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(title="Transformation", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig("colored_cluster_spread.png", dpi=300, bbox_inches="tight")
    print("Saved colored cluster spread plot to colored_cluster_spread.png")
    plt.close()    

def analyze_embeddings(data_dir='./transformed_dataset'):
    print("Loading models...")
    device, clip_model, clip_processor, dinov2_model, dinov2_processor = load_models()
    
    images = []
    labels = []
    clip_embeddings = []
    dinov2_embeddings = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return False
    
    files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.JPEG', '.png'))]
    if not files:
        print(f"No image files found in {data_dir}")
        return False
        
    print(f"Processing {len(files)} images...")
    for img_file in tqdm(files):
        if img_file.startswith('original_'):
            transform_type = 'original'
        else:
            parts = img_file.split('_')
            transform_type = parts[0] if parts else 'unknown'
        
        img_path = os.path.join(data_dir, img_file)
        try:
            clip_emb, dinov2_emb = get_embeddings(img_path, device, clip_model, clip_processor, dinov2_model, dinov2_processor)
            
            images.append(img_path)
            labels.append(transform_type)
            clip_embeddings.append(clip_emb)
            dinov2_embeddings.append(dinov2_emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if not clip_embeddings or not dinov2_embeddings:
        print("No embeddings were successfully extracted")
        return False
    
    clip_embeddings = np.vstack(clip_embeddings)
    dinov2_embeddings = np.vstack(dinov2_embeddings)
    
    print(f"CLIP embedding shape: {clip_embeddings.shape}")
    print(f"DINOv2 embedding shape: {dinov2_embeddings.shape}")
    
    print("\nEvaluating clustering quality...")
    clip_metrics = evaluate_clustering(clip_embeddings, labels, "CLIP")
    dinov2_metrics = evaluate_clustering(dinov2_embeddings, labels, "DINOv2")
    
    metrics_df = pd.DataFrame({
        'Metric': list(clip_metrics.keys()),
        'CLIP': list(clip_metrics.values()),
        'DINOv2': list(dinov2_metrics.values()),
        'Better Model': ['—'] * len(clip_metrics)
    })
    
    for idx, metric in enumerate(metrics_df['Metric']):
        clip_val = metrics_df.loc[idx, 'CLIP']
        dino_val = metrics_df.loc[idx, 'DINOv2']
        
        if np.isnan(clip_val) or np.isnan(dino_val):
            metrics_df.loc[idx, 'Better Model'] = 'N/A'
        elif metric in ['davies_bouldin', 'intra_cluster_distance']:
            if clip_val < dino_val:
                metrics_df.loc[idx, 'Better Model'] = 'CLIP'
            else:
                metrics_df.loc[idx, 'Better Model'] = 'DINOv2'
        else:
            if clip_val > dino_val:
                metrics_df.loc[idx, 'Better Model'] = 'CLIP'
            else:
                metrics_df.loc[idx, 'Better Model'] = 'DINOv2'
    
    metrics_df['% Improvement'] = ['—'] * len(metrics_df)
    for idx, metric in enumerate(metrics_df['Metric']):
        clip_val = metrics_df.loc[idx, 'CLIP']
        dino_val = metrics_df.loc[idx, 'DINOv2']
        
        if np.isnan(clip_val) or np.isnan(dino_val) or clip_val == 0:
            metrics_df.loc[idx, '% Improvement'] = 'N/A'
        elif metric in ['davies_bouldin', 'intra_cluster_distance']:
            improvement = (clip_val - dino_val) / clip_val * 100
            metrics_df.loc[idx, '% Improvement'] = f"{improvement:.2f}%"
        else:
            improvement = (dino_val - clip_val) / clip_val * 100
            metrics_df.loc[idx, '% Improvement'] = f"{improvement:.2f}%"
    
    model_counts = metrics_df['Better Model'].value_counts()
    dino_wins = model_counts.get('DINOv2', 0)
    clip_wins = model_counts.get('CLIP', 0)
    
    print("\n========== CLUSTERING METRICS COMPARISON ==========")
    print(metrics_df.to_string(index=False))
    print("\n==================================================")
    print(f"DINOv2 better on {dino_wins} metrics")
    print(f"CLIP better on {clip_wins} metrics")
    
    metrics_df.to_csv('clustering_metrics_comparison.csv', index=False)
    print("Saved metrics to clustering_metrics_comparison.csv")
    
    print("Generating t-SNE visualizations...")
    clip_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1)).fit_transform(clip_embeddings)
    dinov2_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1)).fit_transform(dinov2_embeddings)

    plot_colored_cluster_spread(clip_2d, dinov2_2d, labels)
    
    print("Creating individual cluster visualizations...")
    clip_avg_distances, dinov2_avg_distances = create_individual_cluster_visualization(
        clip_2d, dinov2_2d, labels, clip_embeddings, dinov2_embeddings
    )
    create_per_transformation_clustering_chart(clip_avg_distances, dinov2_avg_distances)

    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    df_clip = pd.DataFrame({
        'x': clip_2d[:, 0],
        'y': clip_2d[:, 1],
        'transformation': labels
    })
    sns.scatterplot(data=df_clip, x='x', y='y', hue='transformation', alpha=0.7, s=100)
    
    metrics_text = "Clustering Metrics:\n"
    for metric, value in clip_metrics.items():
        if not np.isnan(value):
            if metric in ['silhouette', 'adjusted_rand_index']:
                metrics_text += f"{metric}: {value:.3f}\n"
    
    plt.title('CLIP Embeddings', fontsize=16)
    plt.figtext(0.25, 0.01, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(title='Transformation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    df_dinov2 = pd.DataFrame({
        'x': dinov2_2d[:, 0],
        'y': dinov2_2d[:, 1],
        'transformation': labels
    })
    sns.scatterplot(data=df_dinov2, x='x', y='y', hue='transformation', alpha=0.7, s=100)
    
    metrics_text = "Clustering Metrics:\n"
    for metric, value in dinov2_metrics.items():
        if not np.isnan(value):
            if metric in ['silhouette', 'adjusted_rand_index']:
                metrics_text += f"{metric}: {value:.3f}\n"
    
    plt.title('DINOv2 Embeddings', fontsize=16)
    plt.figtext(0.75, 0.01, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(title='Transformation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to embedding_comparison.png")
    plt.close()
    
    create_metrics_comparison_chart(clip_metrics, dinov2_metrics)
    
    return True

def create_metrics_comparison_chart(clip_metrics, dinov2_metrics):
    higher_better_metrics = ['silhouette', 'calinski_harabasz', 'adjusted_rand_index', 'inter_cluster_distance']
    lower_better_metrics = ['davies_bouldin', 'intra_cluster_distance']
    
    higher_better_data = []
    for metric in higher_better_metrics:
        if metric in clip_metrics and metric in dinov2_metrics:
            if not np.isnan(clip_metrics[metric]) and not np.isnan(dinov2_metrics[metric]):
                higher_better_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'CLIP': clip_metrics[metric],
                    'DINOv2': dinov2_metrics[metric]
                })
    
    lower_better_data = []
    for metric in lower_better_metrics:
        if metric in clip_metrics and metric in dinov2_metrics:
            if not np.isnan(clip_metrics[metric]) and not np.isnan(dinov2_metrics[metric]):
                lower_better_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'CLIP': clip_metrics[metric],
                    'DINOv2': dinov2_metrics[metric]
                })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    if higher_better_data:
        df_higher = pd.DataFrame(higher_better_data)
        df_higher_melted = pd.melt(df_higher, id_vars=['Metric'], var_name='Model', value_name='Score')
        
        sns.barplot(data=df_higher_melted, x='Metric', y='Score', hue='Model', ax=ax1)
        ax1.set_title('Higher is Better Metrics', fontsize=16)
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        for i, metric in enumerate(df_higher['Metric']):
            clip_val = df_higher.iloc[i]['CLIP']
            dino_val = df_higher.iloc[i]['DINOv2']
            if clip_val > 0:
                improvement = (dino_val - clip_val) / clip_val * 100
                x_pos = i
                y_pos = max(clip_val, dino_val) * 1.05
                ax1.text(x_pos, y_pos, f"{improvement:.1f}%", ha='center', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No valid higher-is-better metrics available', 
                 ha='center', va='center', transform=ax1.transAxes)
    
    if lower_better_data:
        df_lower = pd.DataFrame(lower_better_data)
        df_lower_melted = pd.melt(df_lower, id_vars=['Metric'], var_name='Model', value_name='Score')
        
        sns.barplot(data=df_lower_melted, x='Metric', y='Score', hue='Model', ax=ax2)
        ax2.set_title('Lower is Better Metrics', fontsize=16)
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        for i, metric in enumerate(df_lower['Metric']):
            clip_val = df_lower.iloc[i]['CLIP']
            dino_val = df_lower.iloc[i]['DINOv2']
            if clip_val > 0: 
                improvement = (clip_val - dino_val) / clip_val * 100
                x_pos = i
                y_pos = max(clip_val, dino_val) * 1.05
                ax2.text(x_pos, y_pos, f"{improvement:.1f}%", ha='center', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No valid lower-is-better metrics available', 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison_chart.png', dpi=300, bbox_inches='tight')
    print("Saved metrics comparison chart to metrics_comparison_chart.png")
    plt.close()

if __name__ == "__main__":
    success = analyze_embeddings()
    if not success:
        print("Failed to analyze embeddings")