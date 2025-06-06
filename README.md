# 🦺 Security-checker Project 🦺

## 🧐 Contexte 

VisionForge AI est une startup ambitieuse dans le secteur de la surveillance intelligente. L'entreprise développe des solutions de computer vision pour différents secteurs : sécurité urbaine, retail, industrie automobile, et santé publique.

Une société cliente a fait appel à VisionForge AI afin de développer une solution permettant de détecter le porte de casques et de gilets de sécurité. 

Vous êtes IA Engineer chez VisionForge AI. On vous a confié la tâche de développer l'application demandée par le client. La détection doit se faire par image et par vidéo.

## 🔍️ Fonctionnalités techniques 
- Interface visuelle de l'application avec Streamlit
- Prise en charge des images et des vidéos par le modèle de détection
- Détection des casques et des gilets de sécurité par le modèle de détection
- Utilisation d'une API pour exposer le modèle de détection

## 🏗️ Structure du projet
```
security-checker/
|--.streamlit/                  # Application streamlit
|       |--01_home.py
|       |--02_image.py
|       |--03_video.py
|       |--04_live.py
        |--app.py
|       |__config.toml
|--images/                      # Images
        |__logo.webp
|--.gitignore
|--docker-compose.yaml          # Fichier de contenaurisation
|--README.md                    # Informations sur le projet
|__requirements.txt             # Dépendances
```


## 🚀 Démarrage rapide 

#### 1. 👥 Cloner le projet
```bash
git clone https://github.com/MichAdebayo/security-checker.git
cd security-checker
```

#### 2. 🐛 Créer un environnement virtuel avec pip et l'activer
```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### 3. ⬇️ Installer les dépendances
```bash
pip install requirements.txt
```

#### 4. 🔥 Lancer l'application
```bash
cd .streamlit
streamlit run app.py
```