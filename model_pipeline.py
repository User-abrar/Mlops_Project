"""
model_pipeline.py
Pipeline ML modulaire : prédiction de la réadmission (<30, >30, NO)
Jeu de données : Diabetes 130-US Hospitals (diabetic_data.csv)
"""

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def groupe_diagnostic(code):
    """Regroupe un code ICD-9 en grande famille de maladies."""
    if pd.isnull(code):
        return "Inconnu"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Autre"
    n = float(code)
    if 390 <= n <= 459 or n == 785:
        return "Circulatoire"
    if 460 <= n <= 519 or n == 786:
        return "Respiratoire"
    if 520 <= n <= 579 or n == 787:
        return "Digestif"
    if int(n) == 250:
        return "Diabete"
    if 800 <= n <= 999:
        return "Blessure"
    if 710 <= n <= 739:
        return "Musculo-squelettique"
    if 580 <= n <= 629 or n == 788:
        return "Genito-urinaire"
    if 140 <= n <= 239:
        return "Tumeurs"
    return "Autre"


def prepare_data(csv_path="diabetic_data.csv", test_size=0.2, random_state=42):
    """
    Charge et prétraite les données.

    Args:
        csv_path (str): chemin du fichier diabetic_data.csv
        test_size (float): part du jeu de test
        random_state (int): graine aléatoire

    Returns:
        X_train, X_test, y_train, y_test, noms_classes (liste des 3 classes)
    """
    # 1) Chargement ("?" = valeur manquante)
    df = pd.read_csv(csv_path, na_values="?", keep_default_na=False, low_memory=False)

    # 2) Nettoyage : doublons, un séjour par patient, patients décédés
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin([11, 19, 20, 21])]
    df = df.reset_index(drop=True)

    # 3) Cible (3 classes) : <30 -> 0, >30 -> 1, NO -> 2
    le_cible = LabelEncoder()
    y = pd.Series(le_cible.fit_transform(df["readmitted"]), name="readmitted")
    noms_classes = list(le_cible.classes_)
    df = df.drop(columns="readmitted")

    # 4) Colonnes inutiles : >50 % de manquants, identifiants, constantes
    cols_vides = df.columns[df.isnull().mean() > 0.5].tolist()
    cols_id = ["encounter_id", "patient_nbr"]
    cols_constantes = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=cols_vides + cols_id + cols_constantes)

    # 5) Diagnostics regroupés + manquants restants
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(groupe_diagnostic)
    df = df.fillna("Inconnu")

    # 6) Encodage des variables catégorielles
    X = df.copy()
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 7) Séparation train / test (stratifiée)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, noms_classes


def train_model(X_train, y_train, n_iter=15, cv=3, random_state=42):
    """
    Entraîne un XGBoost multiclasse réglé par RandomizedSearchCV.

    Returns:
        best_model : meilleur modèle XGBoost entraîné
    """
    param_grid = {
        "learning_rate": [0.2, 0.1, 0.05, 0.03, 0.01],
        "max_depth": [2, 3, 4, 5, 6, 7],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.5, 1, 2],
        "reg_alpha": [0, 1, 10],
        "reg_lambda": [1, 5, 10],
    }
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
        ),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    print("Meilleurs paramètres :", search.best_params_)
    print("Accuracy (validation croisée) : {:.3f}".format(search.best_score_))
    return search.best_estimator_


def evaluate_model(model, X_test, y_test, noms_classes):
    """
    Évalue le modèle sur le jeu de test.

    Returns:
        dict : accuracy, f1_macro, auc_ovr
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    resultats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr"),
    }
    print("Accuracy :", round(resultats["accuracy"], 3))
    print("F1 macro :", round(resultats["f1_macro"], 3))
    print("AUC (OvR):", round(resultats["auc_ovr"], 3))
    print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))
    print(
        "\n",
        classification_report(
            y_test, y_pred, target_names=noms_classes, zero_division=0
        ),
    )
    return resultats


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print("Modèle sauvegardé :", path)


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(path)
    print("Modèle chargé :", path)
    return model
