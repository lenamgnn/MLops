import pandas as pd
import os
import sys

# Ajouter le répertoire racine MLops au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Importer le module data_loader
from data.data_loader import load_data


def drop_unnecessary_columns(df, columns):
    """
    Supprime les colonnes inutiles d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        columns (list): Liste des colonnes à supprimer.

    Returns:
        pd.DataFrame: DataFrame sans les colonnes spécifiées.
    """
    return df.drop(columns=columns, errors='ignore')


def handle_missing_values(df, strategy='median', columns=None):
    """
    Gère les valeurs manquantes dans un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        strategy (str): Stratégie d'imputation ('mean', 'median', 'mode').
        columns (list): Liste des colonnes à traiter. Si None, toutes les colonnes numériques sont traitées.

    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes imputées.
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns

    for col in columns:
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_categorical_features(df, columns):
    """
    Encode les colonnes catégoriques spécifiées en utilisant des encodages numériques.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        columns (list): Liste des colonnes catégoriques à encoder.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes catégoriques encodées.
    """
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df


def create_age_groups(df):
    """
    Crée une nouvelle colonne "AgeGroup" pour catégoriser les âges en groupes : enfant, adulte, senior.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne "Age".

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne "AgeGroup".
    """
    bins = [0, 18, 60, 120]
    labels = ['Enfant', 'Adulte', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df


def clean_data(df):
    """
    Nettoie les données Titanic :
    - Supprime les colonnes inutiles
    - Gère les valeurs manquantes
    - Encode les colonnes catégoriques
    - Crée une colonne "AgeGroup" pour les groupes d'âge

    Args:
        df (pd.DataFrame): Données brutes.

    Returns:
        pd.DataFrame: Données nettoyées.
    """
    # Supprimer les colonnes inutiles
    columns_to_drop = ['Cabin', 'Ticket']
    df = drop_unnecessary_columns(df, columns_to_drop)

    # Gérer les valeurs manquantes
    df = handle_missing_values(df, strategy='median', columns=['Age', 'Fare'])

    # Encoder les colonnes catégoriques
    categorical_columns = ['Sex', 'Embarked']
    df = encode_categorical_features(df, categorical_columns)

    # Créer des groupes d'âge
    df = create_age_groups(df)

    return df


if __name__ == "__main__":
    # Exemple d'utilisation
    # Utilisation d'un chemin relatif au lieu du chemin absolu
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    file_path = os.path.join(BASE_DIR, "data/raw/train.csv")
    
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # Charger les données
    df = pd.read_csv(file_path)

    print("Données avant nettoyage :")
    print(df.head())

    df_cleaned = clean_data(df)

    print("Données après nettoyage :")
    print(df_cleaned.head())
