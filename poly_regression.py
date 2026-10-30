import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(RegressorMixin, BaseEstimator):

    def __init__(self, degree=2, alpha=0):
        self.degree = degree
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if self.alpha < 0:
            raise ValueError("Alpha ne sme biti negativna.")

        Phi = np.vander(X, self.degree + 1, increasing=True)

        # Matrica za kaznjavanje koeficijenata
        D = np.eye(self.degree + 1)
        D[0, 0] = 0  # Ne kaznjavamo slobodni clan w0

        # Prosirujemo sistem da ukljucimo regularizaciju
        A = np.vstack([Phi, np.sqrt(self.alpha) * D])
        b = np.concatenate([y, np.zeros(self.degree + 1)])

        # Skaliranje kolona radi preciznijeg racunanja
        skala = np.linalg.norm(A, axis=0)
        skala[skala == 0] = 1

        rezultat = np.linalg.lstsq(A / skala, b, rcond=None)[0]
        self.coef_ = rezultat / skala

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        Phi = np.vander(X, self.degree + 1, increasing=True)

        return Phi @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        predikcije = self.predict(X)

        return np.mean((y - predikcije) ** 2)