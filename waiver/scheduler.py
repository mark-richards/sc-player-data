"""
scheduler.py — Runs the waiver newsletter pipeline on a schedule.

Schedule: Every Monday at 07:00 AEST (UTC+10 / Australia/Melbourne).
AFL rounds typically end Sunday night; running Monday morning gives a full
day to review picks before the Tuesday 4:00 AM waiver deadline.

Recommended on Windows: use register_waiver_schedule.bat (Windows Task
Scheduler) instead of keeping this process alive. Task Scheduler fires
even after reboots and handles missed runs automatically.

Usage (APScheduler fallback)
-----
  py waiver/scheduler.py

  # Or run immediately once (for testing) then keep scheduler alive:
  py waiver/scheduler.py --run-now

The process is designed to be kept alive by a process supervisor
(e.g. pm2, supervisord, or Windows Task Scheduler via the .bat file).
"""

import argparse
import logging
import sys

log = logging.getLogger("waiver.scheduler")


def _build_scheduler():
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        log.error(
            "APScheduler is not installed. Run: pip install apscheduler"
        )
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Australia/Melbourne")

    trigger = CronTrigger(
        day_of_week="mon",
        hour=7,
        minute=0,
        timezone="Australia/Melbourne",
    )
    scheduler.add_job(
        _run_job,
        trigger=trigger,
        id="waiver_newsletter",
        name="Weekly Waiver Newsletter — Monday 7 AM AEST",
        max_instances=1,
        coalesce=True,   # if a run was missed, fire once (don't catch up)
    )

    return scheduler


def _run_job():
    """Called by the scheduler. Delegates to the main pipeline."""
    log.info("Scheduler triggered — running waiver newsletter pipeline.")
    try:
        # Import here so the scheduler process itself stays lean
        from run_waiver_newsletter import run_pipeline
        run_pipeline(dry_run=False)
    except Exception as exc:
        log.exception("Pipeline failed during scheduled run: %s", exc)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run the Waiver Newsletter scheduler (Monday 8 PM AEST)."
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute the pipeline immediately before starting the scheduler loop.",
    )
    args = parser.parse_args()

    if args.run_now:
        log.info("--run-now flag set: executing pipeline immediately.")
        _run_job()

    scheduler = _build_scheduler()

    try:
        jobs = scheduler.get_jobs()
        for job in jobs:
            log.info(
                "Scheduled: '%s' — next run: %s",
                job.name,
                job.next_run_time,
            )
    except Exception:
        pass

    log.info(
        "Scheduler started. Waiting for next Monday 07:00 AEST. "
        "Press Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
