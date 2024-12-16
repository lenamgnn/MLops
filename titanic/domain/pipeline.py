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

