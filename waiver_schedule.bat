@echo off
REM ================================================================
REM  Waiver Wire Newsletter — Round Schedule
REM  Auto-generated 2026-07-27 07:06 by waiver/fixture_schedule.py
REM  Source: fixturedownload.com/results/afl-2026
REM
REM  Run as Administrator to register all scheduled tasks.
REM  Re-run this file whenever the newsletter flags it has changed.
REM
REM  Rounds scheduled:
REM    Rd 21 — last game Sun 02 Aug 2026 16:40 AEST → newsletter Mon 03 Aug 2026 @ 07:00 AEST
REM    Rd 22 — last game Sun 09 Aug 2026 19:20 AEST → newsletter Mon 10 Aug 2026 @ 07:00 AEST
REM    Rd 23 — last game Sat 15 Aug 2026 14:00 AEST → newsletter Sun 16 Aug 2026 @ 07:00 AEST
REM    Rd 24 — last game Sat 22 Aug 2026 14:00 AEST → newsletter Sun 23 Aug 2026 @ 07:00 AEST
REM ================================================================

echo Registering Waiver Wire Newsletter tasks...

REM Remove all existing WaiverWire_Rd* tasks (silently)
schtasks /delete /tn "WaiverWire_Rd01" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd02" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd03" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd04" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd05" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd06" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd07" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd08" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd09" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd10" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd11" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd12" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd13" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd14" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd15" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd16" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd17" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd18" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd19" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd20" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd21" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd22" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd23" /f 2>nul
schtasks /delete /tn "WaiverWire_Rd24" /f 2>nul

REM Create tasks for future rounds only
REM Round 21 — Sun 02 Aug last game → newsletter Mon 03 Aug @ 07:00
schtasks /create /tn "WaiverWire_Rd21" /tr "\"G:\My Drive\projects\sc-player-data\run_waiver_newsletter.bat\"" /sc ONCE /sd 03/08/2026 /st 07:00 /ru "%USERNAME%" /rl HIGHEST /f

REM Round 22 — Sun 09 Aug last game → newsletter Mon 10 Aug @ 07:00
schtasks /create /tn "WaiverWire_Rd22" /tr "\"G:\My Drive\projects\sc-player-data\run_waiver_newsletter.bat\"" /sc ONCE /sd 10/08/2026 /st 07:00 /ru "%USERNAME%" /rl HIGHEST /f

REM Round 23 — Sat 15 Aug last game → newsletter Sun 16 Aug @ 07:00
schtasks /create /tn "WaiverWire_Rd23" /tr "\"G:\My Drive\projects\sc-player-data\run_waiver_newsletter.bat\"" /sc ONCE /sd 16/08/2026 /st 07:00 /ru "%USERNAME%" /rl HIGHEST /f

REM Round 24 — Sat 22 Aug last game → newsletter Sun 23 Aug @ 07:00
schtasks /create /tn "WaiverWire_Rd24" /tr "\"G:\My Drive\projects\sc-player-data\run_waiver_newsletter.bat\"" /sc ONCE /sd 23/08/2026 /st 07:00 /ru "%USERNAME%" /rl HIGHEST /f


if %ERRORLEVEL% EQU 0 (
    echo Tasks registered successfully.
) else (
    echo Some tasks may have failed — check output above.
)
pause
