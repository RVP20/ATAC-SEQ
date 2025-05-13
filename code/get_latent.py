import numpy as np
import scanpy as sc
from muon import atac as ac
from scipy.optimize import nnls


def pca(atac, n_comps = 50):
    sc.pp.scale(atac)
    sc.tl.pca(atac, n_comps = n_comps)
    atac.obsm['X_pca'] = atac.obsm['X_pca']
    return atac

def lsi(atac, n_comps=50):
    ac.tl.lsi(atac, n_comps = n_comps)
    atac.obsm['X_lsi'] = atac.obsm['X_lsi'][:,1:]
    return atac


def vanilla_nnmf(atac, n_comps = 50,k_max=100, eps=1e-4):
    ## Apply the vanilla NMF algorithm
    V = atac.X
    m, n = V.shape
    W = np.random.rand(m, n_comps)
    H = np.random.rand(n_comps, n)

    for k in range(k_max):
        W_old = W.copy()
        for i in range(m):
            W[i, :], _ = nnls(H.T, V[i, :])
        for j in range(n):
            H[:, j], _ = nnls(W, V[:, j])
        ## Normalize the columns of W and the rows of H
        
        V_approx = W @ H
        error = np.linalg.norm(V - V_approx, 'fro') / np.linalg.norm(V, 'fro')
        print(f"Iteration {k+1}, Relative Error: {error:.6f}")
        if error < eps:
            break
        if np.linalg.norm(W - W_old, 'fro') < 1e-6:
            print("Converged")
            break

    atac.obsm['X_nnmf_vanilla'] = W
    return atac

def sparse_nnmf(atac, n_comps = 50, beta=0.1, eta=0.1, k_max=100, eps=1e-4):

    ## Apply the sparse NMF algorithm
    A = atac.X
    m, n = A.shape
    W = np.random.rand(m, n_comps)
    H = np.random.rand(n_comps, n)  # shape n x k so that WH^T ~ A

    e_row = np.ones((1, n_comps))
    I_k = np.eye(n_comps)

    for k in range(k_max):
        # --- Update H ---
        W_aug = np.vstack([W, np.sqrt(beta) * e_row])
        A_aug = np.vstack([A, np.zeros((1, n))])
        for j in range(n):
            H[:, j], _ = nnls(W_aug, A_aug[:, j])

        # --- Update W ---
        H_aug = np.hstack([H, np.sqrt(eta) * I_k]).T
        A_T_aug = np.vstack([A.T, np.zeros((n_comps, m))])
        for i in range(m):
            W[i, :], _ = nnls(H_aug, A_T_aug[:, i])

        # --- Compute relative reconstruction error ---
        A_approx = W @ H
        rel_err = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        print(f"Iteration {k+1}: rel error = {rel_err:.6f}")
        if rel_err < eps:
            break
    atac.obsm['X_nnmf_sparse'] = W
    return atac







