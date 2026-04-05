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

## Loading Race Data from Backups

StreamRecorder saves daily `.sql.gz` backups to `../StreamRecorder1/backups/`.
The full pipeline to get training-ready data is:

**Backup .sql.gz** --> **MySQL** (coldData + hotDataRefactored) --> **Parquet** (data/processed/)

There are three ways to run this pipeline:

### Option 1: Restore Wizard (recommended)

1. Start the UI (`start-ui.bat`)
2. Go to **Admin** page
3. Click **Open Restore Wizard**
4. The wizard scans the backup folder and shows available dates
5. Select the date(s) you want, click **Restore**
6. The wizard handles decompression, MySQL restore, and Parquet extraction automatically
7. Progress is shown in real-time via WebSocket

### Option 2: Batch scripts

From the repo root:

```bash
# Restore a single date
./data/restore-backups.sh 2026-04-02

# Restore multiple dates
./data/restore-backups.sh 2026-04-01 2026-04-02

# Restore all dates not yet extracted
./data/restore-backups.sh all
```

On Windows CMD (not Git Bash):
```cmd
data\restore-backups.bat 2026-04-02
data\restore-backups.bat all
```

### Option 3: Manual commands

```bash
# Windows (Git Bash) — adjust date as needed
MYSQL="/c/Program Files/MySQL/MySQL Server 9.6/bin/mysql.exe"

# 1. Restore coldData
"$MYSQL" -u root -p -e "DROP DATABASE IF EXISTS coldData; CREATE DATABASE coldData;"
gzip -dc ../StreamRecorder1/backups/coldData-2026-04-02_223000.sql.gz | "$MYSQL" -u root -p coldData

# 2. Restore hotData
"$MYSQL" -u root -p -e "DROP DATABASE IF EXISTS hotDataRefactored; CREATE DATABASE hotDataRefactored;"
gzip -dc ../StreamRecorder1/backups/hotData-2026-04-02_223000.sql.gz | "$MYSQL" -u root -p hotDataRefactored

# 3. Extract Parquet (with venv activated)
python -c "
from data.extractor import DataExtractor
import yaml, datetime
cfg = yaml.safe_load(open('config.yaml'))
DataExtractor(cfg).extract_date(datetime.date.fromisoformat('2026-04-02'))
"
```

### Backup file format

Backups are pairs of gzip-compressed SQL dumps:
- `coldData-YYYY-MM-DD_HHMMSS.sql.gz` — race metadata, results, weather, runner info
- `hotData-YYYY-MM-DD_HHMMSS.sql.gz` — tick-level market snapshots

Multiple backups per date may exist (e.g. mid-day + end-of-day). The wizard and
batch scripts automatically pick the **latest** backup per date.

## Tests

```bash
pytest                          # unit tests (skips integration)
pytest -m integration           # integration tests (needs MySQL)
cd frontend && npm test         # frontend unit tests
```
