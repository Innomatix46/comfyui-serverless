# 🚀 ComfyUI Serverless API - Quick Start (5 Minuten)

**Die schnellste Art, die API zum Laufen zu bekommen!**

---

## ⚡ Blitzschneller Start

### 1️⃣ Docker installieren
- **Windows/Mac:** https://www.docker.com/products/docker-desktop/
- **Linux:** `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`

### 2️⃣ Projekt klonen
```bash
git clone https://github.com/dein-repo/comfyui-serverless.git
cd comfyui-serverless
```

### 3️⃣ Alles mit einem Befehl starten
```bash
# Setup ausführen:
chmod +x scripts/setup.sh && ./scripts/setup.sh

# Docker Services starten:
docker-compose -f docker-compose.test.yml up -d

# API Server starten:
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Testen
- **API:** http://localhost:8000/health
- **Docs:** http://localhost:8000/docs
- **Storage:** http://localhost:9001 (Login: test-access-key / test-secret-key)

---

## 🧪 Funktionstest

```bash
# Schnelltest ausführen:
python3 test-quick.py

# Volltest (10 Sekunden):
python3 test-docker-integration.py
```

**Ergebnis:** Sollte "🎉 ALL TESTS PASSED!" zeigen.

---

## 🎯 Das war's!

Deine ComfyUI Serverless API läuft jetzt auf:
- **API Server:** http://localhost:8000
- **Dokumentation:** http://localhost:8000/docs
- **Admin Panel:** http://localhost:8000/admin

**Für Details siehe:** [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md)

---

## 🆘 Probleme?

```bash
# Alles zurücksetzen:
docker-compose -f docker-compose.test.yml down --volumes
docker system prune -af
rm -rf venv/
# Dann wieder bei Schritt 3 anfangen
```

**🎉 Happy Coding!**