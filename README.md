# rl-betfair

Reinforcement-learning agent for Betfair horse-racing markets.

## Prerequisites

- **Python 3.12+** with a virtual environment at `.venv/`
- **Node.js 20+** (for the Angular frontend)
- **MySQL 9.x** running on `localhost:3306`
- Database credentials in `.env` (see `.env.example`)

## Quick Start

### Windows

Double-click or run from the repo root:

```
start-ui.bat
```

This opens two console windows — one for the API, one for the Angular dev server.

### Linux / macOS

```bash
./start-ui.sh
```

Both servers run in the same terminal. Press `Ctrl+C` to stop.

### Manual Start

**1. API backend** (from repo root, with venv activated):

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

uvicorn api.main:app --reload --port 8000
```

**2. Angular frontend** (in a second terminal):

```bash
cd frontend
npm start
```

### URLs

| Service  | URL                        |
|----------|----------------------------|
| UI       | http://localhost:4200       |
| API      | http://localhost:8000       |
| API docs | http://localhost:8000/docs  |

The Angular dev server proxies `/api/*` requests to the backend automatically.

## Database Backups

Compressed SQL backups are stored in `../StreamRecorder1/backups/`. To restore:

```bash
# Windows (Git Bash) — adjust date as needed
MYSQL="/c/Program Files/MySQL/MySQL Server 9.6/bin/mysql.exe"
"$MYSQL" -u root -p -e "DROP DATABASE IF EXISTS colddata; CREATE DATABASE colddata;"
gzip -dc ../StreamRecorder1/backups/coldData-2026-04-01_223000.sql.gz | "$MYSQL" -u root -p colddata

"$MYSQL" -u root -p -e "DROP DATABASE IF EXISTS hotdatarefactored; CREATE DATABASE hotdatarefactored;"
gzip -dc ../StreamRecorder1/backups/hotData-2026-04-01_223000.sql.gz | "$MYSQL" -u root -p hotdatarefactored
```

## Tests

```bash
pytest                          # unit tests (skips integration)
pytest -m integration           # integration tests (needs MySQL)
cd frontend && npm test         # frontend unit tests
```
