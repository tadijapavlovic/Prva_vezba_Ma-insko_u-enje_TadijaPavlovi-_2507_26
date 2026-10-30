import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

podaci = pd.read_csv("podaci.csv")

X = podaci[["X"]]
y = podaci["Y"]

stepeni = [1, 2, 3, 5, 9]
vrednosti_alpha = [0, 1, 1000]
nove_padavine = [600, 900, 1300, 1600]

for stepen in stepeni:
    for alpha in vrednosti_alpha:
        model = PolynomialRegression(degree=stepen, alpha=alpha)
        model.fit(X, y)

        mse = model.score(X, y)
        greska = mse / 2

        kazna = alpha * np.sum(model.coef_[1:] ** 2) / (2 * len(y))
        ukupna_greska = greska + kazna

        print("\n---------------------------")
        print("Stepen:", stepen, "| Alpha:", alpha)
        print("Koeficijenti:", model.coef_)
        print("MSE:", mse)
        print("Err bez kazne:", greska)
        print("Funkcija cilja sa kaznom:", ukupna_greska)

        predikcije = model.predict(nove_padavine)

        for padavine, biomasa in zip(nove_padavine, predikcije):
            print(f"{padavine} mm -> {biomasa:.4f} t/ha")


# MATEMATICKO RESENJE:
# Minimizujemo:
# J(w) = (||y - Phi @ w||² + alpha * ||D @ w||²) / (2N)
#
# Izvod po w:
# grad J = (Phi.T @ (Phi @ w - y) + alpha * D.T @ D @ w) / N
#
# Iz uslova grad J = 0 dobijamo:
# (Phi.T @ Phi + alpha * D.T @ D) @ w = Phi.T @ y
#
# Ako je matrica invertibilna:
# w = inv(Phi.T @ Phi + alpha * D.T @ D) @ Phi.T @ y
#
# U kodu isti problem resavamo prosirenim sistemom i lstsq,
# umesto racunanjem inverzne matrice.


# ZAKLJUCAK:
# Alpha = 0 daje rezultate bez regularizacije.
# Regularizacija ogranicava velicinu kaznjenih koeficijenata.
# Greska na obuci moze porasti, a predikcije mogu biti umerenije.
# Veca alpha ne garantuje bolje predikcije na novim podacima.
# X ostaje u milimetrima; uticaj kazne zavisi i od jedinica X