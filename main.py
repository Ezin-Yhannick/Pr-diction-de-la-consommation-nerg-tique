import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# 1. CHARGEMENT DU CSV
# ===============================
csv_path = r"C:\Users\HP\Prediction_Consommation_Electrique\src\data.csv"
df = pd.read_csv(csv_path)

# Vérifie les premières lignes
print(df.head())

# ===============================
# 2. INDEX DATETIME
# ===============================
# On détecte automatiquement la colonne date
date_cols = [col for col in df.columns if "date" in col.lower()]
if not date_cols:
    raise ValueError("Aucune colonne 'date' détectée dans le CSV")
date_col = date_cols[0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)

# ===============================
# 3. NETTOYAGE
# ===============================
# Suppression des doublons
df = df[~df.index.duplicated()]

# Interpolation des valeurs manquantes
if "consommation" not in df.columns:
    raise ValueError("Le CSV doit contenir une colonne 'consommation'")
df["consommation"] = df["consommation"].interpolate()

# ===============================
# 4. FEATURE ENGINEERING
# ===============================
df["heure"] = df.index.hour
df["jour_semaine"] = df.index.dayofweek
df["mois"] = df.index.month
df["weekend"] = df["jour_semaine"].isin([5,6]).astype(int)

# Détecter le pas de temps en heures
pas_heures = (df.index[1] - df.index[0]).total_seconds() / 3600

# Lags : 1 pas et 1 semaine
df["lag_1"] = df["consommation"].shift(1)
df["lag_7"] = df["consommation"].shift(int(24*7/pas_heures))

# Moyenne glissante sur 24h
df["mean_24h"] = df["consommation"].rolling(int(24/pas_heures)).mean()

# Supprimer les lignes avec NaN générés par lags et rolling
df.dropna(inplace=True)

# ===============================
# 5. X / y
# ===============================
X = df.drop(columns=["consommation"])
y = df["consommation"]

# ===============================
# 6. DÉCOUPAGE TEMPOREL
# ===============================
train_end = "2025-09-30"
test_start = "2025-10-01"
test_end = "2025-12-31"

train = df.loc[:train_end]
test  = df.loc[test_start:test_end]

X_train = train.drop(columns=["consommation"])
y_train = train["consommation"]

X_test = test.drop(columns=["consommation"])
y_test = test["consommation"]

# ===============================
# 7. MODÉLISATION (XGBoost)
# ===============================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 8. PRÉDICTION
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 9. ÉVALUATION
# ===============================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# ===============================
# 10. VISUALISATION
# ===============================
plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test, label="Consommation réelle")
plt.plot(y_test.index, y_pred, label="Consommation prédite")
plt.legend()
plt.title("Consommation Réelle vs Prédite")
plt.show()
