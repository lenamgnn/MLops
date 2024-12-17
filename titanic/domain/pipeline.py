
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



def prepare_data(data, target_column='Survived', is_train=True, test_size=0.2, random_state=42):
    """
    Prépare les données pour l'entraînement ou le test.
    Accepte un chemin de fichier ou un DataFrame comme entrée.

    Args:
        data (str or pd.DataFrame): Chemin vers les données ou DataFrame.
        target_column (str): Nom de la colonne cible (utilisé uniquement pour les données d'entraînement).
        is_train (bool): Indique si les données sont pour l'entraînement ou le test.
        test_size (float): Proportion des données pour le test (utilisé pour les données d'entraînement).
        random_state (int): Graine pour la reproductibilité du split.

    Returns:
        tuple: 
            - Si is_train=True: (X_train, X_test, y_train, y_test)
            - Si is_train=False: (X_test,)
    """
    # Charger les données si un chemin est fourni
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("L'argument 'data' doit être un chemin vers un fichier ou un DataFrame.")
    
    # Cas des données d'entraînement
    if is_train:
        # Extraire la cible et les caractéristiques
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Identifier les colonnes catégoriques et numériques
        categorical_features = X.select_dtypes(include=['category', 'object']).columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Définir le préprocesseur
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        # Split en jeu d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Appliquer les transformations
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        # Convertir en DataFrame avec noms de colonnes
        X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
        X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

        return X_train, X_test, y_train, y_test

    # Cas des données de test (aucune cible à extraire)
    else:
        X_test = df.copy()
        
        # Identifier les colonnes catégoriques et numériques
        categorical_features = X_test.select_dtypes(include=['category', 'object']).columns
        numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        
        # Définir le préprocesseur
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        # Appliquer les transformations
        X_test_transformed = preprocessor.fit_transform(X_test)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())
        
        return X_test_transformed,



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

