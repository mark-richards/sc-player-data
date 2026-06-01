@echo off
REM Waiver Wire Newsletter — scheduled launcher
REM Registered with Windows Task Scheduler: Monday 7:00 AM AEST
REM To re-register: run register_waiver_schedule.bat as Administrator

cd /d "g:\My Drive\projects\sc-player-data"
.venv\Scripts\python.exe run_waiver_newsletter.py >> logs\waiver_scheduler.log 2>&1
