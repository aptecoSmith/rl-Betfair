@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  restore-backups.bat — Restore StreamRecorder backups + extract Parquet
REM
REM  ⚠ KNOWN LIMITATION (2026-04-07):
REM  This .bat does NOT yet have the multi-snapshot merge fix that
REM  the .sh version has. StreamRecorder wipes runnerdescription on
REM  every restart, so any single backup file may be empty even
REM  though earlier backups for the same date contain rich data
REM  (see StreamRecorder1/bugs.md B1).
REM
REM  PREFER restore-backups.sh (Git Bash) until this .bat is
REM  updated to merge per-date snapshots with INSERT IGNORE the
REM  way the .sh version does.
REM
REM  Usage:
REM    restore-backups.bat 2026-04-02
REM    restore-backups.bat 2026-04-02 2026-04-03
REM    restore-backups.bat all          (restores all dates not yet extracted)
REM
REM  Prerequisites:
REM    - MySQL 9.6 installed at default location
REM    - gzip available (Git Bash / Git for Windows)
REM    - Python venv at ..\.venv with dependencies installed
REM ============================================================

set "MYSQL=C:\Program Files\MySQL\MySQL Server 9.6\bin\mysql.exe"
set "GZIP=gzip"
set "BACKUP_DIR=%~dp0..\..\StreamRecorder1\backups"
set "PROCESSED_DIR=%~dp0processed"
set "REPO_ROOT=%~dp0.."

REM Check MySQL exists
if not exist "%MYSQL%" (
    echo ERROR: MySQL not found at %MYSQL%
    echo Edit this script to set the correct path.
    exit /b 1
)

REM Check backup dir exists
if not exist "%BACKUP_DIR%" (
    echo ERROR: Backup directory not found at %BACKUP_DIR%
    exit /b 1
)

if "%~1"=="" (
    echo Usage: restore-backups.bat ^<date^> [date2 ...]
    echo        restore-backups.bat all
    echo.
    echo Examples:
    echo   restore-backups.bat 2026-04-02
    echo   restore-backups.bat 2026-04-01 2026-04-02
    echo   restore-backups.bat all
    exit /b 1
)

REM Prompt for MySQL password
set /p "DB_PASS=MySQL root password: "

REM Collect dates to process
set "DATES="
if /i "%~1"=="all" (
    echo Scanning for all available backup dates...
    for /f "tokens=*" %%f in ('dir /b "%BACKUP_DIR%\coldData-*.sql.gz" 2^>nul') do (
        for /f "tokens=2 delims=-" %%a in ("%%~nf") do (
            REM Extract date portion (YYYY-MM-DD) from filename
        )
    )
    REM Simpler approach: list cold files, extract dates
    for /f "delims=" %%f in ('dir /b "%BACKUP_DIR%\coldData-*.sql.gz" 2^>nul ^| sort') do (
        set "fname=%%~nf"
        REM coldData-2026-04-02_223000.sql  -> extract 2026-04-02
        set "datepart=!fname:coldData-=!"
        set "datepart=!datepart:~0,10!"
        if not exist "%PROCESSED_DIR%\!datepart!.parquet" (
            echo   Found new date: !datepart!
            set "DATES=!DATES! !datepart!"
        )
    )
) else (
    :parse_args
    if "%~1"=="" goto :done_args
    set "DATES=!DATES! %~1"
    shift
    goto :parse_args
    :done_args
)

if "!DATES!"=="" (
    echo No dates to restore.
    exit /b 0
)

echo.
echo Dates to restore:!DATES!
echo.

for %%d in (!DATES!) do (
    echo ============================================================
    echo  Restoring %%d
    echo ============================================================

    REM Find latest coldData file for this date
    set "COLD_FILE="
    for /f "delims=" %%f in ('dir /b /o-n "%BACKUP_DIR%\coldData-%%d*.sql.gz" 2^>nul') do (
        if not defined COLD_FILE set "COLD_FILE=%%f"
    )

    REM Find latest hotData file for this date
    set "HOT_FILE="
    for /f "delims=" %%f in ('dir /b /o-n "%BACKUP_DIR%\hotData-%%d*.sql.gz" 2^>nul') do (
        if not defined HOT_FILE set "HOT_FILE=%%f"
    )

    if not defined COLD_FILE (
        echo   WARNING: No coldData backup found for %%d, skipping.
        goto :next_date
    )
    if not defined HOT_FILE (
        echo   WARNING: No hotData backup found for %%d, skipping.
        goto :next_date
    )

    echo   Cold: !COLD_FILE!
    echo   Hot:  !HOT_FILE!

    REM Drop and recreate coldData
    echo   [1/4] Recreating coldData database...
    "%MYSQL%" -u root -p%DB_PASS% -e "DROP DATABASE IF EXISTS coldData; CREATE DATABASE coldData;"
    if errorlevel 1 (
        echo   ERROR: Failed to recreate coldData
        goto :next_date
    )

    REM Restore coldData
    echo   [2/4] Restoring coldData (this may take a moment)...
    %GZIP% -dc "%BACKUP_DIR%\!COLD_FILE!" | "%MYSQL%" -u root -p%DB_PASS% coldData
    if errorlevel 1 (
        echo   ERROR: Failed to restore coldData
        goto :next_date
    )

    REM Drop and recreate hotDataRefactored
    echo   [3/4] Recreating hotDataRefactored and restoring...
    "%MYSQL%" -u root -p%DB_PASS% -e "DROP DATABASE IF EXISTS hotDataRefactored; CREATE DATABASE hotDataRefactored;"
    if errorlevel 1 (
        echo   ERROR: Failed to recreate hotDataRefactored
        goto :next_date
    )

    %GZIP% -dc "%BACKUP_DIR%\!HOT_FILE!" | "%MYSQL%" -u root -p%DB_PASS% hotDataRefactored
    if errorlevel 1 (
        echo   ERROR: Failed to restore hotData
        goto :next_date
    )

    REM Extract Parquet
    echo   [4/4] Extracting Parquet files...
    pushd "%REPO_ROOT%"
    call .venv\Scripts\activate.bat
    python -c "from data.extractor import DataExtractor; import yaml; cfg = yaml.safe_load(open('config.yaml')); DataExtractor(cfg).extract_date(__import__('datetime').date.fromisoformat('%%d'))"
    if errorlevel 1 (
        echo   ERROR: Parquet extraction failed for %%d
    ) else (
        echo   Done: %%d
    )
    popd

    :next_date
)

echo.
echo ============================================================
echo  All done.
echo ============================================================
endlocal
