@echo off
REM Registers the Waiver Wire Newsletter with Windows Task Scheduler.
REM Run this file once as Administrator (right-click → Run as administrator).
REM
REM Schedule: Every Monday at 07:00 (7:00 AM local time / AEST)
REM           — the morning after AFL rounds typically end Sunday night.
REM           Runs even if the machine was off at trigger time (run missed task).

set TASK_NAME=WaiverWireNewsletter
set SCRIPT_PATH=g:\My Drive\projects\sc-player-data\run_waiver_newsletter.bat

echo Registering Task Scheduler task: %TASK_NAME%

schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "\"%SCRIPT_PATH%\"" ^
  /sc WEEKLY ^
  /d MON ^
  /st 07:00 ^
  /ru "%USERNAME%" ^
  /rl HIGHEST ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Task registered successfully.
    echo Next run: next Monday at 07:00
    echo To view: Task Scheduler ^> Task Scheduler Library ^> WaiverWireNewsletter
    echo To run now: schtasks /run /tn "%TASK_NAME%"
    echo To remove: schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo.
    echo Registration failed. Make sure you are running as Administrator.
)
pause
