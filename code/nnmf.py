import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from muon import atac as ac
import os
from scipy.optimize import nnls

def nnmf_anls(atac, rank, max_iter=100, tol=1e-4, verbose=False):
    """
    Perform Non-negative Matrix Factorization using Alternating Nonnegative Least Squares (ANLS).
    
    Parameters:
        V (ndarray): Input non-negative matrix (m x n)
        rank (int): Rank for factorization
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance based on reconstruction error
        verbose (bool): Print loss per iteration
    
    Returns:
        W (ndarray): Basis matrix (m x r)
        H (ndarray): Coefficient matrix (r x n)
    """
    if "counts" not in atac.layers:
        atac.layers["counts"] = atac.X
    else:
        atac.X = atac.layers["counts"]
    ac.pp.tfidf(atac, scale_factor=1e4)
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
    print("Number of highly variable regions: ", np.sum(atac.var.highly_variable))
    atac_hvg = atac[:, atac.var.highly_variable]
    V = atac_hvg.X.toarray()
    m, n = V.shape
    W = np.random.rand(m, rank)
    H = np.random.rand(rank, n)

    for iter_num in range(max_iter):
        # Update W: Solve m independent NNLS problems
        for i in range(m):
            W[i, :], _ = nnls(H.T, V[i, :])
        
        # Update H: Solve n independent NNLS problems
        for j in range(n):
            H[:, j], _ = nnls(W, V[:, j])

        # Compute approximation error
        V_approx = W @ H
        error = np.linalg.norm(V - V_approx, 'fro') / np.linalg.norm(V, 'fro')

        if verbose:
            print(f"Iteration {iter_num+1}, Relative Error: {error:.6f}")
        
        if error < tol:
            break

    atac.obsm['X_nnmf'] = W
    return atac

import numpy as np
from scipy.optimize import nnls

def sparse_nmf_anls(A, rank, beta=0.1, eta=0.1, max_iter=100, tol=1e-4, verbose=False):
    """
    Sparse NMF using Alternating Nonnegative Least Squares with augmented formulation.
    
    Parameters:
        A (ndarray): Input non-negative matrix (m x n)
        rank (int): Rank for factorization
        beta (float): Sparsity regularization on H
        eta (float): Frobenius regularization on W
        max_iter (int): Maximum number of iterations
        tol (float): Relative error tolerance
        verbose (bool): If True, print convergence info
    
    Returns:
        W, H: Factor matrices (W: m x rank, H: n x rank)
    """
    m, n = A.shape
    W = np.random.rand(m, rank)
    H = np.random.rand(n, rank)  # shape n x k so that WH^T ~ A

    e_row = np.ones((1, rank))
    I_k = np.eye(rank)

    for it in range(max_iter):
        # --- Update H ---
        W_aug = np.vstack([W, np.sqrt(beta) * e_row])
        A_aug = np.vstack([A, np.zeros((1, n))])
        for j in range(n):
            H[j, :], _ = nnls(W_aug, A_aug[:, j])

        # --- Update W ---
        H_aug = np.vstack([H, np.sqrt(eta) * I_k])
        A_T_aug = np.vstack([A.T, np.zeros((rank, m))])
        for i in range(m):
            W[i, :], _ = nnls(H_aug, A_T_aug[:, i])

        # --- Compute relative reconstruction error ---
        A_approx = W @ H.T
        rel_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        if verbose:
            print(f"Iter {it+1}: rel error = {rel_err:.6f}")

        if rel_err < tol:
            break

    return W, H

