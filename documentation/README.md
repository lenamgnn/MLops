MLops/
│
├── data/                          		# Contient les données brutes et les scripts liés aux données
│   ├── raw/                       		# Données brutes
│   ├── processed/                 		# Données nettoyées/préparées
│   └── data_loader.py             		# Script de chargement des données
│
├── documentation/                 		# Documentation du projet
│   └── README.md                  		# Tutoriels, explications, etc.
│
├── titanic/                       		# Module principal pour le projet Titanic
│   ├── application/               		# Scripts exécutables
│   │   └── notebooks/             	# Notebooks spécifiques à l'application
│   │       └── titanic_eda_demo.ipynb	# Notebook d'exploration des données
│   │
│   ├── domain/                    		# Contient les pipelines et les modèles
│   │   ├── pipeline.py            	# Pipeline pour préparation et modélisation
│   │   ├── models.py              	# Entraînement et sauvegarde des modèles
│   │   └── evaluation.py          	# Évaluation des modèles
│   │
│   └── infrastructure/            		# Gestion des données et infrastructure
│        └── cleaning.py            	# Script de nettoyage des données
│
├── tests/                         		# Scripts de tests unitaires
│   ├── test_pipeline.py           		# Tests pour le pipeline
│   └── test_cleaning.py           		# Tests pour le nettoyage des données
│
├── requirements.txt               		# Liste des bibliothèques utilisées
└── .gitignore                     		# Exclusions Git
