"""
gsheet_updater.py — Uploads waiver analysis to Google Sheets.

Reuses the service-account authentication pattern from upload_to_gsheet.py.
Writes three named sections to the "Waiver Targets" worksheet:
  - Ceiling Chasers
  - Ruck Radar
  - Drop Zone
"""

import logging
from pathlib import Path

import pandas as pd

from waiver.config import GSHEET_NAME, SERVICE_ACCOUNT_FILE, WORKSHEET_NAME

log = logging.getLogger(__name__)

# Columns written to the sheet (display order)
_SHEET_COLUMNS = [
    "name",
    "pos_bucket",
    "team",
    "effective_avg",
    "ceiling_index",
    "cv",
    "tpor",
    "flex_bonus",
    "trade_status",
    "data_source",
]
_HEADERS = [
    "Player", "Pos", "Team", "Avg", "CI%", "CV", "TPOR",
    "Flex Bonus", "Status", "Data Source",
]


def _get_client():
    """Authenticate with Google Sheets using a service account file."""
    try:
        import gspread
    except ImportError:
        log.error("gspread is not installed. Run: pip install gspread")
        return None

    creds_path = Path(__file__).resolve().parent.parent / SERVICE_ACCOUNT_FILE
    if not creds_path.exists():
        log.error("Service account file not found: %s", creds_path)
        return None
    try:
        return gspread.service_account(filename=str(creds_path))
    except Exception as exc:
        log.error("Google authentication failed: %s", exc)
        return None


def _open_worksheet(gc, spreadsheet_name: str, worksheet_name: str):
    """Open (or create) the target worksheet."""
    import gspread

    try:
        spreadsheet = gc.open(spreadsheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        log.warning("Spreadsheet '%s' not found. Creating it.", spreadsheet_name)
        spreadsheet = gc.create(spreadsheet_name)

    try:
        return spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        log.info("Worksheet '%s' not found. Creating it.", worksheet_name)
        return spreadsheet.add_worksheet(title=worksheet_name, rows="500", cols="20")


def _extract_section(
    metrics_df: pd.DataFrame, section: str
) -> pd.DataFrame:
    """Filter and sort the metrics table for a given newsletter section."""
    df = metrics_df.copy()

    if section == "ceiling_chasers":
        df = df[df["is_free_agent"]].sort_values(
            ["ceiling_index", "flex_bonus"], ascending=[False, False]
        ).head(5)

    elif section == "ruck_radar":
        df = df[df["is_free_agent"] & (df["pos_bucket"] == "RUC")].sort_values(
            "tpor", ascending=False
        )

    elif section == "drop_zone":
        df = df[df["is_on_field"]].copy()
        df["_composite"] = df["tpor"] + df["ceiling_index"] * 0.5
        df = df.sort_values("_composite").head(3)
        df = df.drop(columns=["_composite"])

    else:
        raise ValueError(f"Unknown section: {section!r}")

    return df


def _to_rows(df: pd.DataFrame) -> list[list]:
    """Convert DataFrame to list-of-lists for gspread upload."""
    rows = []
    for _, r in df.iterrows():
        row = []
        for col in _SHEET_COLUMNS:
            val = r.get(col, "")
            if isinstance(val, float):
                val = round(val, 2)
            elif isinstance(val, list):
                val = str(val)
            row.append(val if val is not None else "")
        rows.append(row)
    return rows


def upload_waiver_targets(
    metrics_df: pd.DataFrame,
    spreadsheet_name: str = GSHEET_NAME,
    worksheet_name: str = WORKSHEET_NAME,
) -> bool:
    """
    Clears and rewrites the Waiver Targets worksheet with all three sections.

    Layout:
      Row 1:   Section header "=== CEILING CHASERS ==="
      Row 2:   Column headers
      Row 3-7: Top 5 FA by CI
      Row 8:   blank
      Row 9:   Section header "=== RUCK RADAR ==="
      ...etc.

    Returns True on success, False on failure.
    """
    gc = _get_client()
    if gc is None:
        return False

    ws = _open_worksheet(gc, spreadsheet_name, worksheet_name)

    data: list[list] = []

    sections = [
        ("ceiling_chasers", "=== CEILING CHASERS (Top FA by CI%) ==="),
        ("ruck_radar",       "=== RUCK RADAR (Available Rucks by TPOR) ==="),
        ("drop_zone",        "=== DROP ZONE (My Most Droppable Starters) ==="),
    ]

    for section_key, section_title in sections:
        data.append([section_title])
        data.append(_HEADERS)

        try:
            section_df = _extract_section(metrics_df, section_key)
            section_rows = _to_rows(section_df)
            data.extend(section_rows)
        except Exception as exc:
            log.error("Error building section %s: %s", section_key, exc)
            data.append([f"Error: {exc}"])

        data.append([""])  # blank separator row

    try:
        ws.clear()
        ws.update(data, "A1")
        log.info("Google Sheets updated successfully: '%s' → '%s'", spreadsheet_name, worksheet_name)
        return True
    except Exception as exc:
        log.error("Failed to update Google Sheets: %s", exc)
        return False
