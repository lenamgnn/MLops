from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import os
def prepare_data(df, target_column='Survived'):
    """
    Prépare les données pour l'entraînement en séparant les features et la cible.
    Applique l'encodage pour les variables catégoriques et la standardisation pour les variables numériques.

    Args:
        df (pd.DataFrame): Données nettoyées prêtes à être traitées.
        target_column (str): Nom de la colonne cible.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Séparer les features (X) et la cible (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identifier les colonnes numériques et catégoriques
    categorical_features = X.select_dtypes(include=['category', 'object']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Définir les transformations pour chaque type de colonne
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Appliquer les transformations
    X_transformed = preprocessor.fit_transform(X)

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Exemple d'utilisation
    cleaned_data_path = "/workspaces/MLOPS/data/processed/cleaned_train.csv"
    df = pd.read_csv(cleaned_data_path)

    print("Données nettoyées avant préparation :")
    print(df.head())

    # Préparer les données
    X_train, X_test, y_train, y_test = prepare_data(df)

    print("Nombre d'échantillons dans l'ensemble d'entraînement :", len(X_train))
    print("Nombre d'échantillons dans l'ensemble de test :", len(X_test))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import os
import sys

# Ajouter le répertoire racine MLops au PYTHONPATH
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importer les modules nécessaires
from titanic.infrastructure.cleaning import clean_and_save_data
from data.data_loader import load_data


def prepare_data(input_path, target_column='Survived'):
    """
    Prépare les données pour l'entraînement en séparant les features et la cible.
    Applique l'encodage pour les variables catégoriques et la standardisation pour les variables numériques.

    Args:
        input_path (str): Chemin vers les données brutes.
        target_column (str): Nom de la colonne cible.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Définir le chemin des données nettoyées
    output_path = os.path.join(project_root, "data/processed/cleaned_train.csv")

    # Nettoyer les données si le fichier nettoyé n'existe pas
    if not os.path.exists(output_path):
        clean_and_save_data(input_path, output_path)

    # Charger les données nettoyées
    df = load_data(output_path)

    # Identifier les colonnes
    X = df.drop(columns=[target_column])
    if target_column is not None : y = df[target_column]

    categorical_features = X.select_dtypes(include=['category', 'object']).columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Définir les transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Appliquer les transformations
    X_transformed = preprocessor.fit_transform(X)

    # Diviser les données en train/test
    if target_column is not None : 
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else : return X_transformed



if __name__ == "__main__":
    # Chemin vers les données brutes
    input_path = os.path.join(project_root, "data/raw/train.csv")

    print("Préparation des données en cours...")

    try:
        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(input_path)

        print("Nombre d'échantillons dans l'ensemble d'entraînement :", len(X_train))
        print("Nombre d'échantillons dans l'ensemble de test :", len(X_test))

    except Exception as e:
        print("Erreur lors de la préparation des données :", e)

