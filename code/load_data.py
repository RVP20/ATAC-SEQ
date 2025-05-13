import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import atac as ac
import os

def load_data(num_input_features = 5000):
    mdata = mu.read_10x_h5(os.path.join("./data/", "filtered_feature_bc_matrix.h5"))
    mdata.var_names_make_unique()
    metadata = pd.read_csv("./metadata.csv", index_col = 0)
    metadata.index = metadata.index + "-1"
    metadata = metadata[~metadata['celltype'].isna()]

    mask = np.isin(mdata.obs_names, metadata.index)


    atac = mdata['atac']
    atac = atac[mask, :]
    atac.obs = atac.obs.join(metadata)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 10)
    atac.layers["counts"] = atac.X

    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5, n_top_genes = num_input_features)
    print("Number of highly variable regions: ", np.sum(atac.var.highly_variable))
    atac = atac[:, atac.var.highly_variable]
    atac.X = atac.X.toarray()
    return atac

    



