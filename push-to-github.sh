#!/bin/bash

# WICHTIG: Ersetze DEIN-GITHUB-USERNAME mit deinem echten GitHub Benutzernamen!
GITHUB_USERNAME="DEIN-GITHUB-USERNAME"

echo "📌 GitHub Repository verknüpfen..."
git remote add origin https://github.com/${GITHUB_USERNAME}/comfyui-serverless.git

echo "📤 Code zu GitHub pushen..."
git push -u origin main

echo "✅ Fertig! Repository ist auf GitHub!"
echo ""
echo "📋 Nächste Schritte:"
echo "1. Gehe zu: https://github.com/${GITHUB_USERNAME}/comfyui-serverless/settings/secrets/actions"
echo "2. Füge diese Secrets hinzu:"
echo "   - DOCKER_USERNAME (dein Docker Hub Benutzername)"
echo "   - DOCKER_PASSWORD (dein Docker Hub Access Token)"
echo ""
echo "3. GitHub Actions wird automatisch das Docker Image bauen!"