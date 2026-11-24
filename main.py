"""
main.py : exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --step prepare
    python main.py --step train
    python main.py --step evaluate
    python main.py --step all
"""

import argparse
import time
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def titre(texte):
    print("\n" + "=" * 60)
    print(f"  {texte}")
    print("=" * 60)


def etape_prepare(data_path):
    titre("ÉTAPE 1 : PRÉPARATION DES DONNÉES")
    t0 = time.time()
    X_train, X_test, y_train, y_test, noms_classes = prepare_data(data_path)
    print(f"Fichier        : {data_path}")
    print(f"Train          : {X_train.shape}")
    print(f"Test           : {X_test.shape}")
    print(f"Classes        : {list(noms_classes)}")
    print(f"Durée          : {time.time() - t0:.1f} s")
    return X_train, X_test, y_train, y_test, noms_classes


def etape_train(X_train, y_train, n_iter, model_path):
    titre("ÉTAPE 2 : ENTRAÎNEMENT (XGBoost + RandomizedSearchCV)")
    t0 = time.time()
    print(f"Combinaisons testées : {n_iter}")
    model = train_model(X_train, y_train, n_iter=n_iter)
    print(f"Entraînement terminé en {time.time() - t0:.1f} s")

    # Affiche les meilleurs hyperparamètres si le modèle est un RandomizedSearchCV
    if hasattr(model, "best_params_"):
        print("Meilleurs paramètres :", model.best_params_)
    if hasattr(model, "best_score_"):
        print(f"Meilleur score CV    : {model.best_score_:.4f}")

    titre("ÉTAPE 3 : SAUVEGARDE DU MODÈLE")
    save_model(model, model_path)
    print(f"Modèle sauvegardé dans : {model_path}")
    return model


def etape_evaluate(model_path, X_test, y_test, noms_classes):
    titre("ÉTAPE 4 : CHARGEMENT DU MODÈLE")
    model = load_model(model_path)
    print(f"Modèle chargé depuis : {model_path}")

    titre("ÉTAPE 5 : ÉVALUATION")
    evaluate_model(model, X_test, y_test, noms_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Réadmission (XGBoost)")
    parser.add_argument(
        "--step",
        required=True,
        choices=["prepare", "train", "evaluate", "all"],
        help="Étape à exécuter",
    )
    parser.add_argument("--data", default="diabetic_data.csv", help="Chemin du CSV")
    parser.add_argument("--model", default="model.joblib", help="Chemin du modèle")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="Nb de combinaisons testées (RandomizedSearchCV)",
    )
    args = parser.parse_args()

    debut = time.time()

    # Les données sont toujours préparées (résultat déterministe : random_state=42)
    X_train, X_test, y_train, y_test, noms_classes = etape_prepare(args.data)

    if args.step == "train":
        etape_train(X_train, y_train, args.n_iter, args.model)

    elif args.step == "evaluate":
        etape_evaluate(args.model, X_test, y_test, noms_classes)

    elif args.step == "all":
        etape_train(X_train, y_train, args.n_iter, args.model)
        etape_evaluate(args.model, X_test, y_test, noms_classes)

    titre(f"PIPELINE TERMINÉ en {time.time() - debut:.1f} s")


if __name__ == "__main__":
    main()
