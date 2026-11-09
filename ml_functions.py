import numpy as np
import pandas as pd

def standardize_data(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std_matrix = (X - X_mean) / X_std
    
    return X_std_matrix, X_mean, X_std

def unstandardize_data(X_std_matrix, X_mean, X_std):
    X_original = (X_std_matrix * X_std) + X_mean
    return X_original

def update_piece_values(W, X_mean, X_std):
    new_values = (W[1:, 0] * X_std) + X_mean
    return new_values

def GradientDescent_polinomial(X, y, max_iterations=100, alpha=1):
    m, n = X.shape # number of samples, number of features
    J = np.zeros(max_iterations)

    # y must be a column vector of shape m x 1
    y = y.reshape(m, 1)
    
    # Create as many parameters as features and initialize them to zero
    w = np.zeros(shape=(n, 1))
    
    # Repeat for max_iterations (it would be nice to also check convergence...)
    for iteration in range(max_iterations):
        grad = np.dot(X.T , (np.dot(X,w) - y)) / m;
        w = w - alpha*grad
        J[iteration] = (((np.dot(X,w) - y)**2) / 2).sum()
    return [w, J]


def GradientDescent_polinomial_withRidge(X, y, max_iterations=100, alpha=1, lam = 1):
    m, n = X.shape # number of samples, number of features
    J = np.zeros(max_iterations)

    # y must be a column vector of shape m x 1
    y = y.reshape(m, 1)

    # Create as many parameters as features and initialize them to zero
    w = np.zeros(shape=(n, 1))

    # Repeat for max_iterations (it would be nice to also check convergence...)
    for iteration in range(max_iterations):
        w_copy = w.copy()
        w_copy[0] = 0 # ignore the bias term w0 in the regularisation
        grad = (X.T @ ((X @ w) - y))/m + 2 * lam * w_copy
        w = w - alpha*grad


        w_copy = w.copy()# needs to be updated again so it calcualtes the loss with the new values not mixing the old values of w with thte news
        w_copy[0] = 0
        J[iteration] = np.sum(((X @ w) - y)**2)/(2*m) + lam * (w_copy.T @ w_copy).item()
    return [w, J]


def GradientDescent_polynomial_withLasso(X, y, max_iterations=100, alpha=1, lam=1):
    m, n = X.shape  # number of samples, number of features
    J = np.zeros(max_iterations)

    # y must be a column vector of shape m x 1
    y = y.reshape(m, 1)

    # Create as many parameters as features and initialize them to zero
    w = np.zeros(shape=(n, 1))

    for iteration in range(max_iterations):
        w_copy = w.copy()
        w_copy[0] = 0  # ignore the bias term w0 in the regularization
        
        # Subgradient of L1 penalty (lam * |w|) is lam * sign(w)
        grad = (X.T @ ((X @ w) - y)) / m + lam * np.sign(w_copy)
        w = w - alpha * grad

        w_copy = w.copy()  # update copy with the new weights for loss calculation
        w_copy[0] = 0
        
        # L1 penalty is lam * sum(|w|)
        J[iteration] = np.sum(((X @ w) - y)**2) / (2 * m) + lam * np.sum(np.abs(w_copy))
        
    return [w, J]


def generation_learning(EXP_PER_GEN):

    df = pd.read_csv('results/experiments_dataset.csv').tail(EXP_PER_GEN)

    X = df[['pawn', 'knight', 'bishop', 'rook', 'queen']].to_numpy()
    y = df[['score']].to_numpy()

    # Normalize each row by its mean so only the relative proportions between pieces matter,
    # centered around 430 because is the average of the base_algorithm. The value doesn't matter as long as it is smaller equal than 2000
    X_norm = X / np.mean(X, axis=1, keepdims=True) * 430

    std_X, mean_X, sigma_X = standardize_data(X_norm)

    m = std_X.shape[0]
    std_X_design = np.hstack([np.ones((m, 1)), std_X])

    # Ridge keeps weights from blowing up since normalizing rows makes the 5 features dependent,
    # and prevents overfitting to noisy game scores on small generations
    W, J = GradientDescent_polinomial_withRidge(std_X_design, y, max_iterations=1000, alpha=0.1, lam=0.01)

    scale = np.mean(mean_X)
    new_piece_values = np.clip(update_piece_values(W, mean_X, sigma_X), 0.0005 * scale, 5.0 * scale)

    return new_piece_values, mean_X, sigma_X
