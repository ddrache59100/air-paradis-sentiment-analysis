#!/bin/bash
cd /home/site/wwwroot

# Installer les dépendances si elles ne sont pas déjà installées
if [ ! -f ".dependencies_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .dependencies_installed
fi

# Démarrer l'application
gunicorn --bind=0.0.0.0 --timeout 600 api:app