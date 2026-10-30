import pandas as pd
from poly_regression import PolynomialRegression

podaci = pd.read_csv("podaci.csv")

X = podaci[["X"]]
y = podaci["Y"]

stepeni = [1, 2, 3, 5, 9]
nove_padavine = [600, 900, 1300, 1600]

for stepen in stepeni:
    model = PolynomialRegression(degree=stepen)
    model.fit(X, y)

    mse = model.score(X, y)
    greska = mse / 2

    print("\n---------------------------")
    print("Stepen polinoma:", stepen)
    print("Koeficijenti:", model.coef_)
    print("Funkcija greske Err:", greska)
    print("MSE:", mse)

    predikcije = model.predict(nove_padavine)

    for padavine, biomasa in zip(nove_padavine, predikcije):
        print(f"{padavine} mm -> {biomasa:.4f} t/ha")


# ZAKLJUCAK:
# Stepen 1 daje pravu i ne prati dobro rast pa pad biomase.
# To je nedovoljna prilagodjenost (underfitting).
#
# Povecanjem stepena smanjuje se greska na podacima za obuku.
# Stepen 9 skoro potpuno prolazi kroz sva merenja, ali za
# 1600 mm predvidja oko 77.19 t/ha, sto je veoma velika vrednost.
# To pokazuje opasnost prekomerne prilagodjenosti (overfitting).
#
# Stepeni 2 i 3 daju umerenije predikcije.
# Bez novih merenja ne znamo koji model ima najmanju test gresku.