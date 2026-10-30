import pandas as pd
from poly_regression import PolynomialRegression

podaci = pd.read_csv("podaci.csv")

X = podaci[["X"]]
y = podaci["Y"]

model = PolynomialRegression(degree=2)

model.fit(X, y)

print("Koeficijenti:", model.coef_)
print("Predikcije:", model.predict(X))
print("MSE:", model.score(X, y))