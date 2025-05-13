
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(atac, key):
    true_labels = atac.obs['celltype']
    predicted_labels = atac.obs[key]
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f"ARI: {ari}, NMI: {nmi}")
    return ari, nmi


def cluster(atac, latent_name = 'pca', resolution = 0.5, n_neighbors = 10, num_clusters = 10):
    if latent_name == 'raw':
        sc.pp.neighbors(atac, use_rep='X', n_neighbors=n_neighbors, key_added = latent_name)
    else:
        sc.pp.neighbors(atac, use_rep=f'X_{latent_name}', n_neighbors=n_neighbors, key_added = latent_name)
    sc.tl.leiden(atac, resolution=resolution, neighbors_key = latent_name, key_added = f'leiden_{latent_name}')
    if num_clusters is not None:
        n_clusters_detected = len(np.unique(atac.obs[f'leiden_{latent_name}']))
        while n_clusters_detected != num_clusters:
            print(f"Number of clusters detected: {n_clusters_detected}, expected: {num_clusters}")
            if n_clusters_detected > num_clusters:
                resolution *= 0.9
            else:
                resolution *= 1.1
            print(f"New resolution: {resolution}")
            sc.tl.leiden(atac, resolution=resolution, neighbors_key = latent_name, key_added = f'leiden_{latent_name}')
            n_clusters_detected = len(np.unique(atac.obs[f'leiden_{latent_name}']))
            
    return atac

def plot_umap(atac, latent_name= 'pca', keys = None):
    sc.tl.umap(atac, neighbors_key = latent_name)
    if keys is None:
        sc.pl.umap(atac, color=f'leiden_{latent_name}')
    else:
        sc.pl.umap(atac, color=keys)


