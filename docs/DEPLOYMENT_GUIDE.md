# ComfyUI Serverless API - Deployment Guide für Anfänger 🚀

**Eine Schritt-für-Schritt Anleitung für das Deployment ohne Vorkenntnisse**

---

## 📋 Was du brauchst (Voraussetzungen)

### 💻 System Requirements
- **Computer:** Windows, macOS oder Linux
- **RAM:** Mindestens 8GB (16GB empfohlen)
- **Speicher:** 20GB freier Speicherplatz
- **Internet:** Stabile Internetverbindung

### 🛠 Software die installiert werden muss
1. **Docker Desktop** - Für Container-Management
2. **Python 3.11+** - Programmiersprache
3. **Git** - Für Code-Download (optional)
4. **Text-Editor** - VS Code empfohlen (optional)

---

## 🔧 SCHRITT 1: Docker installieren

### Windows:
1. Gehe zu https://www.docker.com/products/docker-desktop/
2. Klicke "Download for Windows"
3. Führe die heruntergeladene .exe Datei aus
4. Folge den Installationsschritten
5. **Wichtig:** Computer nach Installation neu starten
6. Docker Desktop starten (sollte automatisch passieren)

### macOS:
1. Gehe zu https://www.docker.com/products/docker-desktop/
2. Klicke "Download for Mac" (Intel oder Apple Silicon)
3. Ziehe Docker.app in den Applications Ordner
4. Starte Docker Desktop aus Applications
5. Erlaube alle Berechtigungen die gefragt werden

### Linux (Ubuntu/Debian):
```bash
# In Terminal eingeben:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Computer neu starten
```

### ✅ Docker testen:
Öffne Terminal/Kommandozeile und gib ein:
```bash
docker --version
docker-compose --version
```
Du solltest Versionsnummern sehen (z.B. "Docker version 24.0.6")

---

## 🐍 SCHRITT 2: Python installieren

### Windows:
1. Gehe zu https://www.python.org/downloads/
2. Klicke "Download Python 3.12.x"
3. **WICHTIG:** Aktiviere "Add Python to PATH" ✅
4. Installiere mit "Install Now"

### macOS:
```bash
# Homebrew installieren (falls noch nicht vorhanden):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python installieren:
brew install python@3.12
```

### Linux:
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

### ✅ Python testen:
```bash
python3 --version
# Sollte zeigen: Python 3.12.x
```

---

## 📁 SCHRITT 3: Projekt herunterladen

### Option A: Mit Git (empfohlen)
```bash
git clone https://github.com/dein-username/comfyui-serverless.git
cd comfyui-serverless
```

### Option B: Als ZIP herunterladen
1. Gehe zur GitHub Seite des Projekts
2. Klicke grünen "Code" Button
3. Klicke "Download ZIP"
4. Entpacke die ZIP-Datei
5. Öffne Terminal im entpackten Ordner

---

## 🚀 SCHRITT 4: Automatisches Setup (Einfachster Weg)

### Für alle Betriebssysteme:

1. **Terminal/Kommandozeile öffnen** im Projekt-Ordner

2. **Setup-Script ausführen:**
```bash
# Ausführungsrechte geben (Linux/Mac):
chmod +x scripts/setup.sh

# Script ausführen:
./scripts/setup.sh
```

**Was passiert automatisch:**
- ✅ Python Virtual Environment wird erstellt
- ✅ Alle Abhängigkeiten werden installiert
- ✅ Konfigurationsdateien werden erstellt
- ✅ Datenbank wird vorbereitet
- ✅ Alle Verzeichnisse werden angelegt

### Windows PowerShell Alternative:
```powershell
# PowerShell als Administrator öffnen
python -m venv venv
.\venv\Scripts\activate
pip install -r src/requirements.txt
pip install -r requirements-dev.txt
```

---

## 🐳 SCHRITT 5: Docker Services starten

### Alle Services mit einem Befehl starten:
```bash
docker-compose -f docker-compose.test.yml up -d
```

**Was wird gestartet:**
- 🗄️ **PostgreSQL** - Datenbank (Port 5432)
- 🏃 **Redis** - Cache System (Port 6379) 
- 📦 **MinIO** - File Storage (Port 9000-9001)
- 🎨 **ComfyUI Mock** - API Simulation (Port 8188)

### ✅ Prüfen ob alles läuft:
```bash
docker-compose -f docker-compose.test.yml ps
```

**Du solltest sehen:**
```
NAME                     STATUS
postgres-1              Up (healthy)
redis-1                  Up (healthy)
minio-1                  Up (healthy)
mock-comfyui-1          Up (healthy)
```

---

## ⚙️ SCHRITT 6: Konfiguration anpassen

### .env Datei bearbeiten:
```bash
# .env Datei öffnen mit einem Text-Editor
notepad .env        # Windows
nano .env           # Linux
open -e .env        # macOS
```

### Wichtige Einstellungen ändern:
```env
# Sicherheit - IMMER ÄNDERN!
SECRET_KEY=DEIN_SUPER_SICHERER_SCHLÜSSEL_HIER

# Datenbank (schon korrekt für Docker)
DATABASE_URL=postgresql://test:test@localhost/test_comfyui

# Redis (schon korrekt für Docker)
REDIS_URL=redis://localhost:6379/0

# Storage (schon korrekt für Docker)
STORAGE_TYPE=s3
AWS_ACCESS_KEY_ID=test-access-key
AWS_SECRET_ACCESS_KEY=test-secret-key
AWS_S3_BUCKET=test-bucket
S3_ENDPOINT=http://localhost:9000

# API Einstellungen
DEBUG=true
HOST=0.0.0.0
PORT=8000
```

**🔒 Sicherheits-Tipp:** Generiere einen starken SECRET_KEY:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 🎯 SCHRITT 7: API Server starten

### Virtual Environment aktivieren:
```bash
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### API Server starten:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### ✅ Erfolg prüfen:
Du solltest sehen:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 🌐 SCHRITT 8: API testen

### 1. Health Check:
Öffne Browser und gehe zu:
```
http://localhost:8000/health
```
Du solltest sehen: `{"status":"ok"}`

### 2. API Dokumentation:
```
http://localhost:8000/docs
```
**Hier findest du:**
- 📖 Alle verfügbaren API Endpoints
- 🧪 Interaktive Test-Oberfläche
- 📋 Request/Response Beispiele

### 3. Admin Interface:
```
http://localhost:8000/admin
```

### 4. Monitoring Dashboard:
```
http://localhost:8000/metrics
```

---

## 🧪 SCHRITT 9: Funktionalität testen

### Automatische Tests ausführen:
```bash
# Einfacher Test:
python test-quick.py

# Vollständige Test-Suite:
python test-docker-integration.py

# Mit pytest:
pytest tests/unit/ -v
```

### Manueller API Test mit curl:
```bash
# Health Check
curl http://localhost:8000/health

# Workflow einreichen (Beispiel)
curl -X POST "http://localhost:8000/workflows/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "nodes": {
        "1": {
          "class_type": "LoadImage", 
          "inputs": {"image": "test.png"}
        }
      }
    },
    "webhook_url": "https://your-webhook.com/callback"
  }'
```

---

## 📊 SCHRITT 10: Monitoring & Logs

### Docker Logs anzeigen:
```bash
# Alle Services:
docker-compose -f docker-compose.test.yml logs

# Einzelner Service:
docker-compose -f docker-compose.test.yml logs redis
docker-compose -f docker-compose.test.yml logs postgres
docker-compose -f docker-compose.test.yml logs mock-comfyui
```

### API Logs anzeigen:
API Logs werden direkt im Terminal angezeigt wo der Server läuft.

### Storage Web Interface (MinIO):
```
http://localhost:9001
```
**Login:** 
- Username: `test-access-key`
- Password: `test-secret-key`

---

## 🚨 Häufige Probleme & Lösungen

### Problem: "Port already in use"
```bash
# Prüfe welche Prozesse die Ports verwenden:
lsof -i :8000
lsof -i :5432
lsof -i :6379

# Beende Docker Services:
docker-compose -f docker-compose.test.yml down

# Starte neu:
docker-compose -f docker-compose.test.yml up -d
```

### Problem: "Permission denied"
```bash
# Linux/macOS - Datei-Rechte korrigieren:
chmod +x scripts/setup.sh
sudo chown -R $USER:$USER venv/
```

### Problem: "Module not found"
```bash
# Virtual Environment aktivieren:
source venv/bin/activate

# Dependencies neu installieren:
pip install -r src/requirements.txt
```

### Problem: Docker startet nicht
1. Docker Desktop neu starten
2. Computer neu starten
3. Docker Desktop Einstellungen prüfen (Resources > Memory > 4GB+)

### Problem: Datenbank Connection Error
```bash
# PostgreSQL Container neu starten:
docker-compose -f docker-compose.test.yml restart postgres

# Status prüfen:
docker-compose -f docker-compose.test.yml ps postgres
```

---

## 🔒 SCHRITT 11: Produktions-Deployment

### Für echte Produktion (nicht nur zum Testen):

1. **Sicherheit erhöhen:**
```bash
# Starken SECRET_KEY generieren:
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# In .env speichern und DEBUG=false setzen
DEBUG=false
```

2. **Echte Datenbank verwenden:**
```env
# PostgreSQL in der Cloud (z.B. AWS RDS):
DATABASE_URL=postgresql://user:password@your-db-host:5432/comfyui_prod
```

3. **SSL/HTTPS aktivieren:**
```bash
# Nginx oder Traefik als Reverse Proxy verwenden
# SSL-Zertifikat von Let's Encrypt holen
```

4. **Echtes ComfyUI installieren:**
- ComfyUI auf einem GPU-Server installieren
- `COMFYUI_API_URL` in .env auf echte URL setzen

---

## 📞 Hilfe & Support

### 🆘 Wenn gar nichts funktioniert:

1. **Alles zurücksetzen:**
```bash
# Docker komplett bereinigen:
docker-compose -f docker-compose.test.yml down --volumes
docker system prune -af

# Virtual Environment löschen:
rm -rf venv/

# Neu anfangen ab Schritt 4
```

2. **Logs sammeln:**
```bash
# Alle wichtigen Logs in eine Datei:
docker-compose -f docker-compose.test.yml logs > debug.log
pip freeze > installed_packages.txt
```

### 📚 Weitere Ressourcen:
- **API Dokumentation:** http://localhost:8000/docs
- **Test Report:** Siehe `test-report.md`
- **GitHub Issues:** Für Bug-Reports und Fragen
- **Discord/Forum:** Community Support

---

## 🎉 Herzlichen Glückwunsch!

Du hast erfolgreich die ComfyUI Serverless API deployed! 

### Was du jetzt hast:
- ✅ Vollständig funktionsfähige API
- ✅ Datenbank für Workflow-Speicherung
- ✅ Cache-System für Performance
- ✅ File Storage für Uploads
- ✅ Monitoring und Logs
- ✅ Interaktive API Dokumentation
- ✅ Umfassende Test-Suite

### Nächste Schritte:
1. 🎨 Echtes ComfyUI anschließen
2. 🌐 Domain und SSL einrichten
3. 📊 Monitoring erweitern
4. 🔐 Sicherheit für Produktion härten
5. 🚀 Skalierung planen

**Happy Coding!** 🎯