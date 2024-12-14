
import pandas as pd
import os

def load_data(file_path):
    """
    Charge un fichier CSV en tant que DataFrame Pandas.

    Args:
        file_path (str): Chemin complet vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier est vide.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Le fichier {file_path} est vide.")

    return df

def save_data(df, output_path):
    """
    Sauvegarde un DataFrame en tant que fichier CSV.

    Args:
        df (pd.DataFrame): DataFrame à sauvegarder.
        output_path (str): Chemin complet pour le fichier de sortie.

    Returns:
        None
    """
    df.to_csv(output_path, index=False)
    print(f"Données sauvegardées avec succès dans {output_path}.")


