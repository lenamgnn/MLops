import pytest
import pandas as pd
import sys
import os


# Ajouter le répertoire racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from titanic.infrastructure.cleaning import clean_data

# Définir une fixture pour charger et préparer sample_data
@pytest.fixture
def sample_data():
    """
    Crée un DataFrame d'exemple basé sur un échantillon réel des données.
    """
    # Charger les données
    train_data = pd.read_csv('data/raw/train.csv')

    # Créer un échantillon
    sample_data = train_data.sample(n=5, random_state=42)

    # Sélectionner les colonnes pertinentes
    columns_to_include = ['Survived', 'Age', 'Fare', 'Sex', 'Embarked']
    sample_data = sample_data[columns_to_include]

    # Remplir les valeurs manquantes
    sample_data['Age'] = sample_data['Age'].fillna(sample_data['Age'].median())
    sample_data['Embarked'] = sample_data['Embarked'].fillna('S')

    return sample_data

def test_clean_data(sample_data: pd.DataFrame):
    """
    Teste la fonction clean_data pour s'assurer que les données sont correctement nettoyées.
    """
    # Nettoyer les données
    cleaned_data = clean_data(sample_data)

    # Vérifications
    # Vérifie que les colonnes inutiles sont supprimées
    assert 'Cabin' not in cleaned_data.columns, "La colonne 'Cabin' devrait être supprimée."
    assert 'Ticket' not in cleaned_data.columns, "La colonne 'Ticket' devrait être supprimée."

    # Vérifie que les valeurs manquantes dans 'Age' et 'Fare' sont imputées
    assert cleaned_data['Age'].isnull().sum() == 0, "Les valeurs manquantes dans 'Age' devraient être imputées."
    assert cleaned_data['Fare'].isnull().sum() == 0, "Les valeurs manquantes dans 'Fare' devraient être imputées."

    # Vérifie que les valeurs manquantes dans 'Embarked' sont imputées
    assert cleaned_data['Embarked'].isnull().sum() == 0, "Les valeurs manquantes dans 'Embarked' devraient être imputées."

    # Vérifie que les colonnes catégoriques sont encodées
    assert cleaned_data['Sex'].dtype.name == 'int8', "La colonne 'Sex' devrait être encodée en valeurs numériques."
    assert cleaned_data['Embarked'].dtype.name == 'int8', "La colonne 'Embarked' devrait être encodée en valeurs numériques."

    print("Test de clean_data réussi !")

if __name__ == "__main__":
    # Exécution directe des tests pour vérifier leur validité
    pytest.main(["-v", __file__])
