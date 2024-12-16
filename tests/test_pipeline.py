import pytest
import pandas as pd
import sys
import os


# Ajouter le répertoire racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from titanic.domain.pipeline import prepare_data

# Définir une fixture pour charger et préparer sample_data
@pytest.fixture
def sample_data():
    """
    Crée et renvoie un échantillon de données pour le test.
    """
    # Charger les données
    train_data = pd.read_csv('data/raw/train.csv')

    # Créer un échantillon
    sample_data = train_data.sample(n=30, random_state=42)

    # Sélectionner les colonnes pertinentes
    columns_to_include = ['Survived', 'Age', 'Fare', 'Sex', 'Embarked']
    sample_data = sample_data[columns_to_include]

    # Remplir les valeurs manquantes
    sample_data['Age'] = sample_data['Age'].fillna(sample_data['Age'].median())
    sample_data['Embarked'] = sample_data['Embarked'].fillna('S')

    return sample_data

def test_prepare_data(sample_data):
    """
    Teste la fonction prepare_data pour s'assurer qu'elle retourne des ensembles correctement préparés.
    """
    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(sample_data, target_column='Survived')

    # Vérifications
    assert X_train.shape[0] > 0, "X_train ne doit pas être vide"
    assert X_test.shape[0] > 0, "X_test ne doit pas être vide"
    assert y_train.shape[0] > 0, "y_train ne doit pas être vide"
    assert y_test.shape[0] > 0, "y_test ne doit pas être vide"

    # Vérifie que le nombre de colonnes correspond aux transformations
    assert X_train.shape[1] == X_test.shape[1], "Les dimensions des colonnes de X_train et X_test doivent correspondre"

    print("Test de prepare_data réussi !")

if __name__ == "__main__":
    # Exécution des tests avec pytest
    pytest.main(["-v", __file__])
