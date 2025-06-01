import numpy as np


def procrustes(X: np.ndarray, Y: np.ndarray, verbose=False, scaling=False):
    assert X.shape[1] == 3
    assert Y.shape[1] == 3
    assert X.shape[1] == Y.shape[1]

    x_centroid = np.mean(X, axis=0)
    y_centroid = np.mean(Y, axis=0)
    X_centered = X - x_centroid
    Y_centered = Y - y_centroid

    # SVD of covariance matrix
    H = X_centered.T @ Y_centered
    U, S, Vh = np.linalg.svd(H)

    # Reflection handling
    det = np.linalg.det(Vh.T @ U.T)
    if det < 0:
        D = np.eye(3)
        D[2, 2] = -1
        R = Vh.T @ D @ U.T
        trace = S[0] + S[1] - S[2]  # Adjust trace for reflection
    else:
        R = Vh.T @ U.T
        trace = S.sum()
        D = np.eye(3)

    # Calculate scale if requested
    s = 1.0
    if scaling:
        s = trace / np.sum(X_centered**2)

    t = y_centroid - s * R @ x_centroid

    if verbose:
        Y_hat = (s * (X @ R.T)) + t
        print(f"Alignment Loss: {np.abs(Y_hat - Y).mean():.6f}")
        if scaling:
            print(f"Estimated scale: {s:.6f}")

    return (s, R, t) if scaling else (R, t)
