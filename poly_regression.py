 # Skaliranje kolona radi preciznijeg racunanja
        skala = np.linalg.norm(Phi, axis=0)
        skala[skala == 0] = 1

        # Racunanje koeficijenata metodom najmanjih kvadrata
        rezultat = np.linalg.lstsq(Phi / skala, y, rcond=None)[0]
        self.coef_ = rezultat / skala

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1)
        Phi = np.vander(X, self.degree + 1, increasing=True)

        return Phi @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        predikcije = self.predict(X)

        return np.mean((y - predikcije) ** 2)