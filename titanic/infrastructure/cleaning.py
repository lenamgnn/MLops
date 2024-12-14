import pandas as pd

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

