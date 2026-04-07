#!/usr/bin/env bash
# ============================================================
#  restore-backups.sh — Restore StreamRecorder backups + extract Parquet
#
#  Usage:
#    ./data/restore-backups.sh 2026-04-02
#    ./data/restore-backups.sh 2026-04-02 2026-04-03
#    ./data/restore-backups.sh all   (restores all dates not yet extracted)
#
#  Run from the repo root, or from the data/ folder.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MYSQL="/c/Program Files/MySQL/MySQL Server 9.6/bin/mysql.exe"
BACKUP_DIR="$REPO_ROOT/../StreamRecorder1/backups"
PROCESSED_DIR="$REPO_ROOT/data/processed"

# Check prerequisites
if [ ! -f "$MYSQL" ]; then
    echo "ERROR: MySQL not found at $MYSQL"
    echo "Edit this script to set the correct path."
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: Backup directory not found at $BACKUP_DIR"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: restore-backups.sh <date> [date2 ...]"
    echo "       restore-backups.sh all"
    echo ""
    echo "Examples:"
    echo "  ./data/restore-backups.sh 2026-04-02"
    echo "  ./data/restore-backups.sh 2026-04-01 2026-04-02"
    echo "  ./data/restore-backups.sh all"
    exit 1
fi

# Prompt for password
read -sp "MySQL root password: " DB_PASS
echo

# Collect dates
DATES=()
if [ "$1" = "all" ]; then
    echo "Scanning for all available backup dates..."
    for f in "$BACKUP_DIR"/coldData-*.sql.gz; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        # Extract date: coldData-2026-04-02_223000.sql.gz -> 2026-04-02
        date_part=$(echo "$fname" | sed -E 's/coldData-([0-9]{4}-[0-9]{2}-[0-9]{2}).*/\1/')
        if [ ! -f "$PROCESSED_DIR/${date_part}.parquet" ]; then
            # Avoid duplicates
            if [[ ! " ${DATES[*]} " =~ " ${date_part} " ]]; then
                DATES+=("$date_part")
                echo "  Found new date: $date_part"
            fi
        fi
    done
else
    DATES=("$@")
fi

if [ ${#DATES[@]} -eq 0 ]; then
    echo "No dates to restore."
    exit 0
fi

echo ""
echo "Dates to restore: ${DATES[*]}"
echo ""

#
# StreamRecorder writes daily-cache tables (runnerdescription,
# Todays_markets, Todays_venues) which it WIPES on every restart
# (StreamRecorder1/Program.cs:189). It snapshots backups frequently,
# so for any given date there are usually multiple backup files —
# some taken when the cache was full, some taken right after a wipe
# when the cache is empty. The OLDEST snapshot of a date often has
# the richest data because it can also still contain runners cached
# from earlier days that hadn't been wiped yet.
#
# Strategy: for each requested date, we MERGE all of that date's
# backup files into one restore. The first backup (oldest) is
# applied with its original DDL (DROP TABLE / CREATE TABLE / INSERT).
# Each subsequent backup has its DDL stripped out, and its INSERTs
# rewritten to INSERT IGNORE so duplicate primary keys (rows already
# loaded by an earlier backup) are silently skipped instead of
# erroring. The end state is the union of all snapshots for that
# date, which is the most complete view we can reconstruct.
#
# See StreamRecorder1/bugs.md B1 for the underlying root cause.

# Strip mysqldump DDL/locks/key-toggles from a decompressed dump and
# convert INSERTs into INSERT IGNORE so we can append safely.
strip_ddl_for_merge() {
    sed -E \
        -e '/^DROP TABLE IF EXISTS /d' \
        -e '/^CREATE TABLE /,/^\) ENGINE=/d' \
        -e '/^LOCK TABLES /d' \
        -e '/^UNLOCK TABLES/d' \
        -e '/^\/\*!40000 ALTER TABLE .* DISABLE KEYS \*\/;/d' \
        -e '/^\/\*!40000 ALTER TABLE .* ENABLE KEYS \*\/;/d' \
        -e 's/^INSERT INTO /INSERT IGNORE INTO /'
}

for d in "${DATES[@]}"; do
    echo "============================================================"
    echo " Restoring $d"
    echo "============================================================"

    # Collect ALL coldData backups for this date, sorted oldest-first
    # so the merge order matches the chronological order they were
    # taken in.  The hot backup is still chosen as just the latest
    # (hot data isn't subject to the same daily-wipe issue).
    mapfile -t COLD_FILES < <(ls -1 "$BACKUP_DIR"/coldData-"${d}"*.sql.gz 2>/dev/null | sort)
    HOT_FILE=$(ls -1 "$BACKUP_DIR"/hotData-"${d}"*.sql.gz 2>/dev/null | sort -r | head -1)

    if [ ${#COLD_FILES[@]} -eq 0 ]; then
        echo "  WARNING: No coldData backup found for $d, skipping."
        continue
    fi
    if [ -z "$HOT_FILE" ]; then
        echo "  WARNING: No hotData backup found for $d, skipping."
        continue
    fi

    echo "  Cold: ${#COLD_FILES[@]} backup(s) to merge:"
    for f in "${COLD_FILES[@]}"; do
        echo "    - $(basename "$f")"
    done
    echo "  Hot:  $(basename "$HOT_FILE")"

    # ── Restore coldData (merged) ────────────────────────────────
    echo "  [1/4] Recreating coldData database..."
    "$MYSQL" -u root -p"$DB_PASS" -e "DROP DATABASE IF EXISTS coldData; CREATE DATABASE coldData;"

    echo "  [2/4] Restoring coldData (merging ${#COLD_FILES[@]} snapshot(s))..."
    # First snapshot: apply intact (DDL + data).
    gzip -dc "${COLD_FILES[0]}" | "$MYSQL" -u root -p"$DB_PASS" coldData
    # Subsequent snapshots: strip DDL, INSERT IGNORE the data.
    for ((i = 1; i < ${#COLD_FILES[@]}; i++)); do
        gzip -dc "${COLD_FILES[$i]}" | strip_ddl_for_merge \
            | "$MYSQL" -u root -p"$DB_PASS" coldData
    done

    # Sanity-check the merged result for the most-likely-empty table.
    rd_count=$("$MYSQL" -u root -p"$DB_PASS" -N -B coldData \
        -e "SELECT COUNT(*) FROM runnerdescription" 2>/dev/null || echo "?")
    echo "    runnerdescription rows after merge: $rd_count"

    # ── Restore hotData ──────────────────────────────────────────
    echo "  [3/4] Recreating hotDataRefactored and restoring..."
    "$MYSQL" -u root -p"$DB_PASS" -e "DROP DATABASE IF EXISTS hotDataRefactored; CREATE DATABASE hotDataRefactored;"
    gzip -dc "$HOT_FILE" | "$MYSQL" -u root -p"$DB_PASS" hotDataRefactored

    # ── Extract Parquet ──────────────────────────────────────────
    echo "  [4/4] Extracting Parquet files..."
    cd "$REPO_ROOT"
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
    python -c "
from data.extractor import DataExtractor
import yaml, datetime
cfg = yaml.safe_load(open('config.yaml'))
DataExtractor(cfg).extract_date(datetime.date.fromisoformat('$d'))
"

    echo "  Done: $d"
done

echo ""
echo "============================================================"
echo " All done."
echo "============================================================"
