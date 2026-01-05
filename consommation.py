import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tkinter as tk
from tkinter import filedialog
import sys
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV 

# ------------------------------------------------------------------------
# 1. CHARGEMENT
# ------------------------------------------------------------------------  
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

print("Sélectionnez vos fichiers CSV...")
chemins_fichiers = filedialog.askopenfilenames(title="Sélectionnez les datasets (CSV)", filetypes=[("Fichiers CSV", "*.csv")])

if not chemins_fichiers:
    sys.exit("Aucun fichier choisi. Fin du programme.")

liste_df = []
for f in chemins_fichiers:
    try:
        temp_df = pd.read_csv(f, sep=None, engine='python', encoding='ISO-8859-1') 
        liste_df.append(temp_df)
    except Exception as e:
        print(f"Erreur sur le fichier {f}: {e}")

df = pd.concat(liste_df, axis=0)

# Identification automatique
col_date = next((c for c in df.columns if any(m in c.lower() for m in ['date', 'heure', 'time'])), df.columns[0])
col_conso = next((c for c in df.columns if any(m in c.lower() for m in ['cons', 'mw', 'load']) and 'prév' not in c.lower()), df.columns[1])

df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
df[col_conso] = pd.to_numeric(df[col_conso], errors='coerce')
df = df.dropna(subset=[col_date, col_conso]).set_index(col_date).sort_index()
df = df[[col_conso]]
df.columns = ['target']
df = df[~df.index.duplicated(keep='first')]
df['target'] = df['target'].interpolate(method='linear')

# ------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Optimisé avec Lags et Cycles)
# ------------------------------------------------------------------------      
def fonctions_avancees(df):
    df = df.copy()
    # Cycles horaires (Sin/Cos pour que minuit soit proche de 23h)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Lags (Retards)
    df['lag_1h'] = df['target'].shift(1)
    df['lag_24h'] = df['target'].shift(24)
    df['lag_7j'] = df['target'].shift(168)
    
    # Moyenne glissante pour la tendance
    df['rolling_mean_24h'] = df['target'].shift(1).rolling(window=24).mean()
    
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df.dropna()

df = fonctions_avancees(df)

#----------------------------------------------------------------------------
# 3. OPTIMISATION DES HYPERPARAMÈTRES ET ENTRAÎNEMENT
#----------------------------------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

split_limit = 8760 if len(df) > 10000 else int(len(df) * 0.2)
X_train, X_test = X.iloc[:-split_limit], X.iloc[-split_limit:]
y_train, y_test = y.iloc[:-split_limit], y.iloc[-split_limit:]

print("\n⚙️  Recherche des meilleurs hyperparamètres (XGBoost)...")
param_grid = {
    'n_estimators': [100, 500, 800],
    'max_depth': [3, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# Recherche aléatoire de la meilleure configuration
random_search = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), 
                                   param_distributions=param_grid, 
                                   n_iter=5, cv=3, verbose=1)
random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_

# Modèle LightGBM (très rapide et complémentaire)
model_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
model_lgb.fit(X_train, y_train)

# Prédiction combinée (Ensemble learning)
preds_total = (best_xgb.predict(X_test) + model_lgb.predict(X_test)) / 2 

#-------------------------------------------------------------------------------
# 4. EXPORTATION DÉTAILLÉE
#-------------------------------------------------------------------------------
df_export = pd.DataFrame({
    'Reel': y_test.values,
    'Prediction': preds_total,
    'Erreur': y_test.values - preds_total
}, index=y_test.index)

df_export.to_csv('predictions_finales_detaillees.csv', index=True)
print(f"✅ Export terminé. Score R² : {r2_score(y_test, preds_total):.4f}")

#------------------------------------------------------------------------------
# 5. VISUALISATION
#------------------------------------------------------------------------------
def afficher_graphique(choix):
    limites = {"1": 168, "2": 2160, "3": len(y_test)}
    n = limites.get(choix, 168)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index[:n], y_test.values[:n], label='Réel', color='blue', alpha=0.7)
    plt.plot(y_test.index[:n], preds_total[:n], label='Prédiction', color='red', linestyle='--')
    plt.title(f"Analyse des prédictions (Zoom : {n} points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

choix = input("\nVisualisation : 1 (Semaine), 2 (Trimestre), 3 (Total) : ")
afficher_graphique(choix)