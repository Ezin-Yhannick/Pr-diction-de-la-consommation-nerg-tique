# Prediction-de-la-consommation-energetique (XGBoost & LightGBM)
Entrainer un modèle IA pour prédire la consommation énergétique d'une zone en fonctions de certaines données 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Regression-orange.svg)](https://scikit-learn.org/)

Ce projet implémente une solution de bout en bout pour la prévision de séries temporelles énergétiques. Grâce à une architecture hybride combinant **XGBoost** et **LightGBM**, le modèle est capable de capturer les saisonnalités complexes de la consommation électrique.

## 🌟 Points Forts
- **Pipeline Automatisé** : Chargement multi-fichiers et détection automatique des colonnes temporelles.
- **Prétraitement Robuste** : Correction des trous temporels par rééchantillonnage et nettoyage des valeurs aberrantes (Outliers) via la méthode IQR.
- **Feature Engineering Cyclique** : Transformation de l'heure en coordonnées sinus/cosinus pour une meilleure perception du temps par l'IA.
- **Optimisation d'Hyperparamètres** : Recherche aléatoire (RandomizedSearch) pour maximiser la précision du modèle XGBoost.

---

## 🛠️ Installation

1. **Cloner le dépôt** :
   ```bash
   git clone 
2. Installer les dépendances : pip install pandas numpy xgboost lightgbm scikit-learn matplotlib tkinter
3. Méthodologie
   1. Préparation des donnéesLe script effectue un nettoyage "industriel" :Resampling ('H') : Garantit une continuité chronologique (indispensable pour les calculs de Lags).Interpolation Linéaire : Comble intelligemment les données manquantes.Dédoublonnage : Sécurise l'index temporel.
   2. Ingénierie des Variables (Features) Pour compenser l'absence de données météo, le modèle s'appuie sur : Lags : Valeurs à $t-1h$, $t-24h$ et $t-168h$ (semaine précédente).Moyennes Mobiles : Capture de la tendance sur les dernières 24 heures.Variables Temporelles : Jour de la semaine, mois, et indicateurs cycliques.
   3. Entraînement & OptimisationLe modèle final est un Ensemble :XGBoost : Optimisé par RandomizedSearchCV (profondeur, taux d'apprentissage, etc.).LightGBM : Utilisé pour sa rapidité et sa capacité de généralisation.Moyenne pondérée : Les deux prédictions sont combinées pour réduire la variance de l'erreur.
4. Évaluation & VisualisationLe script calcule dynamiquement les métriques pour trois horizons temporels :Semaine (168h) Trimestre (2160h) Totalité des données de testMétrique Description MAEErreur Moyenne Absolue (précision directe en MW/kW). MSE Erreur Quadratique Moyenne (pénalise les grands écarts).R² Capacité du modèle à expliquer la variance (cible : > 0.90).
5.  Utilisation
   Exécutez le script :
     ```bash
      python consommation.py
     ```bash
   Sélectionnez vos fichiers CSV via l'interface.
   Observez les métriques de performance s'afficher dans la console.
   Visualisez les courbes de prédiction selon l'échelle souhaitée.Retrouvez vos résultats détaillés dans predictions_consommations.csv.
