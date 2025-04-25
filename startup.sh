#!/bin/bash
cd /home/site/wwwroot

# Installer les dépendances au démarrage
# En cas d'échec précédent ou de mise à jour des requirements
pip install -r requirements.txt

# Démarrer l'application
gunicorn --bind=0.0.0.0:8000 --timeout 600 api:app
