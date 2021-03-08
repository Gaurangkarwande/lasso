import numpy as np
import time
import copy


def normalization(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma
    return X_norm / np.linalg.norm(X_norm, axis=0)


def predict(X, beta):
    return np.dot(X, beta)


class CoordinateDescent:
    def __init__(self, lamda, max_iters):
        self.max_iters = max_iters
        self.lamda = lamda

    def fit(self, X, y):
        m, n = X.shape
        beta = np.ones((n, 1))
        t = time.time()
        m, n = X.shape
        cost = 20000
        for iter in range(self.max_iters):
            beta_old = copy.deepcopy(beta)
            non_zero_ids = np.where(beta != 0)[0]
            for j in non_zero_ids:
                y_pred = predict(X, beta)
                Xj = X[:, j][:, np.newaxis]
                ols = np.sum(Xj * (y - y_pred + beta[j] * Xj))
                beta[j] = self.soft_thresholding(ols)
            print('\n Finished iteration ', iter)
            print(' Number of non-zeros beta are ', np.sum((beta != 0)))
            print(' Cost is ', self.find_cost(X, y, beta))
            if np.all(abs(beta - beta_old) < 0.0001):
                print('Reached convergence. Exiting loop')
                break
        mse = np.mean(np.square(y - predict(X, beta)))
        elapsed = time.time() - t
        return beta, mse, elapsed

    def find_cost(self, X, y, beta):
        final_prediction = np.dot(X, beta)
        J = 1 / 2 * np.mean(np.square(y - final_prediction)) + self.lamda * np.linalg.norm(beta, 1)
        return J

    def soft_thresholding(self, ols):
        if ols < -self.lamda:
            s = ols + self.lamda
        elif ols > self.lamda:
            s = ols - self.lamda
        else:
            s = 0
        return s
