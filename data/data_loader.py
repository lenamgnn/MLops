import pandas as pd
import os

# Récupérer le chemin du dossier contenant ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(file_path):
    """
    Charge un fichier CSV en tant que DataFrame Pandas.

    Args:
        file_path (str): Chemin relatif ou absolu vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est vide.
    """
    full_path = os.path.join(BASE_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Le fichier {full_path} n'existe pas.")

    df = pd.read_csv(full_path)

    if df.empty:
        raise ValueError(f"Le fichier {full_path} est vide.")

    return df

def save_data(df, output_path):
    """
    Sauvegarde un DataFrame en tant que fichier CSV.

    Args:
        df (pd.DataFrame): DataFrame à sauvegarder.
        output_path (str): Chemin relatif ou absolu pour le fichier de sortie.

    Returns:
        None
    """
    full_path = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Crée les dossiers si nécessaire
    df.to_csv(full_path, index=False)
    print(f"Données sauvegardées avec succès dans {full_path}.")


def load_train_test_data(train_path, test_path):
    """
    Charge les fichiers CSV de train et de test.

    Args:
        train_path (str): Chemin vers le fichier d'entraînement (train.csv).
        test_path (str): Chemin vers le fichier de test (test.csv).

    Returns:
        tuple: (DataFrame pour train, DataFrame pour test)
    """
    print("Chargement des données d'entraînement...")
    train_df = load_data(train_path)

    print("Chargement des données de test...")
    test_df = load_data(test_path)

    print("Données chargées avec succès.")
    return train_df, test_df

if __name__ == "__main__":
    # Chemins relatifs des fichiers
    train_path = "raw/train.csv"
    test_path = "raw/test.csv"

    # Charger les données
    train_df, test_df = load_train_test_data(train_path, test_path)

    # Exemple d'accès aux premières lignes des DataFrames
    print("Aperçu des données d'entraînement :")
    print(train_df.head())

    print("Aperçu des données de test :")
    print(test_df.head())

    # Sauvegarder un exemple après modification
    processed_path = "processed/processed_train.csv"
    save_data(train_df, processed_path)
