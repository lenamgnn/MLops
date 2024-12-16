from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest sur les données d'entraînement.

    Args:
        X_train (array-like): Données d'entraînement.
        y_train (array-like): Cibles d'entraînement.

    Returns:
        model: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path="models/random_forest.pkl"):
    """
    Sauvegarde un modèle entraîné dans un fichier.

    Args:
        model: Modèle entraîné.
        file_path (str): Chemin pour sauvegarder le modèle.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Modèle sauvegardé dans {file_path}.")

def load_model(file_path="models/random_forest.pkl"):
    """
    Charge un modèle sauvegardé depuis un fichier.

    Args:
        file_path (str): Chemin du modèle sauvegardé.

    Returns:
        model: Modèle chargé.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    return joblib.load(file_path)

def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle sur les données de test.

    Args:
        model: Modèle entraîné.
        X_test (array-like): Données de test.
        y_test (array-like): Cibles de test.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Exemple d'exécution (remplacez par les chemins et données réels)
    from pipeline import prepare_data

    # Préparer les données
    input_path = "data/raw/train.csv"
    X_train, X_test, y_train, y_test = prepare_data(input_path)

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Sauvegarder le modèle
    save_model(model)

    # Charger et évaluer le modèle
    loaded_model = load_model()
    evaluate_model(loaded_model, X_test, y_test)
