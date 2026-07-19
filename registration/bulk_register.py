"""
registration/bulk_register.py
-----------------------------
Register MANY stakeholders at once from an Excel (.xlsx) or CSV file.

Your sheet must have these column headers in row 1 (any order,
case-insensitive; extra columns are ignored):

    photo   -> image filename or path  (e.g. 0001.jpg  or  stakeholder/0001.jpg)
    name    -> full name               (e.g. Sita Sharma)
    role    -> Student / Faculty / Staff / Authorized
    uid     -> OPTIONAL unique id (e.g. S001). If the column is missing or a
               cell is empty, the uid is taken from the photo filename
               (0001.jpg -> "0001").

Usage (from the project root, venv active):

    python registration/bulk_register.py --sheet stakeholders.xlsx --photos-dir stakeholder

    --sheet       path to your .xlsx or .csv file            (required)
    --photos-dir  folder that holds the photos; used to resolve photo
                  cells that are bare filenames               (default: ".")
    --dry-run     validate everything and show what WOULD be registered,
                  without touching the database

Rows that fail (missing photo, no detectable face, bad role) are reported
at the end — the rest are still registered, so one bad row never blocks
the whole batch. Re-running is safe: existing UIDs are updated, not duplicated.
"""

import argparse
import os
import sys

# Allow running this file directly: make project-root imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.logger import get_logger

log = get_logger()

VALID_ROLES = {"student": "Student", "faculty": "Faculty",
               "staff": "Staff", "authorized": "Authorized"}
REQUIRED_COLUMNS = {"photo", "name", "role"}


def load_sheet(sheet_path):
    """Read .xlsx or .csv into a DataFrame with lower-cased, trimmed headers."""
    if not os.path.isfile(sheet_path):
        raise FileNotFoundError(f"Sheet not found: {sheet_path}")
    ext = os.path.splitext(sheet_path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(sheet_path)
    elif ext == ".csv":
        df = pd.read_csv(sheet_path)
    else:
        raise ValueError(f"Unsupported sheet type '{ext}' (use .xlsx or .csv)")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Sheet is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}. "
            "Row 1 must contain headers: photo, name, role (uid optional).")
    return df


def clean_cell(value):
    """Trim a cell; return '' for NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def resolve_photo(photo_cell, photos_dir, sheet_dir):
    """
    Turn a photo cell into an existing absolute path, trying in order:
      1. the cell as given (absolute or relative to where you run the command)
      2. inside --photos-dir
      3. relative to the sheet's own folder
    Returns the first path that exists, else None.
    """
    candidates = [
        photo_cell,
        os.path.join(photos_dir, photo_cell),
        os.path.join(sheet_dir, photo_cell),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def parse_rows(df, photos_dir, sheet_dir):
    """
    Validate every row. Returns (valid, errors) where valid is a list of
    dicts {uid, name, role, photo} and errors is a list of readable strings.
    Excel row numbers in messages account for the header row (data starts at 2).
    """
    valid, errors, seen_uids = [], [], set()
    for i, row in df.iterrows():
        rownum = i + 2
        name = clean_cell(row.get("name"))
        role_raw = clean_cell(row.get("role"))
        photo_cell = clean_cell(row.get("photo"))
        uid = clean_cell(row.get("uid")) if "uid" in df.columns else ""

        if not (name or role_raw or photo_cell):
            continue  # fully empty row — skip silently

        if not name:
            errors.append(f"Row {rownum}: 'name' is empty")
            continue
        role = VALID_ROLES.get(role_raw.lower())
        if role is None:
            errors.append(f"Row {rownum} ({name}): role '{role_raw}' invalid "
                          f"— use Student/Faculty/Staff/Authorized")
            continue
        if not photo_cell:
            errors.append(f"Row {rownum} ({name}): 'photo' is empty")
            continue
        photo = resolve_photo(photo_cell, photos_dir, sheet_dir)
        if photo is None:
            errors.append(f"Row {rownum} ({name}): photo not found: "
                          f"'{photo_cell}' (looked in --photos-dir "
                          f"'{photos_dir}' and next to the sheet)")
            continue
        if not uid:  # derive uid from the filename: 0001.jpg -> 0001
            uid = os.path.splitext(os.path.basename(photo))[0]
        if uid in seen_uids:
            errors.append(f"Row {rownum} ({name}): duplicate uid '{uid}' "
                          f"in the sheet — give each person a unique uid")
            continue
        seen_uids.add(uid)
        valid.append({"uid": uid, "name": name, "role": role, "photo": photo})
    return valid, errors


def bulk_register(sheet_path, photos_dir=".", dry_run=False):
    """Main driver. Returns (registered_count, error_messages)."""
    df = load_sheet(sheet_path)
    sheet_dir = os.path.dirname(os.path.abspath(sheet_path))
    valid, errors = parse_rows(df, photos_dir, sheet_dir)

    print(f"\nSheet: {sheet_path}")
    print(f"Valid rows: {len(valid)}   Problem rows: {len(errors)}\n")
    for person in valid:
        print(f"  {person['uid']:<10} {person['name']:<28} "
              f"{person['role']:<10} {os.path.basename(person['photo'])}")

    if dry_run:
        print("\n--dry-run: nothing was registered.")
        return 0, errors

    if not valid:
        print("Nothing to register.")
        return 0, errors

    # Import ML modules only when actually registering (keeps --dry-run fast).
    from registration import register_stakeholder

    registered = 0
    for person in valid:
        try:
            sid = register_stakeholder.register_from_images(
                person["uid"], person["name"], person["role"], person["photo"])
            if sid is not None:
                registered += 1
            else:
                errors.append(f"{person['name']} ({person['uid']}): no "
                              f"detectable face in the photo — use a clearer, "
                              f"front-facing image")
        except Exception as exc:
            errors.append(f"{person['name']} ({person['uid']}): {exc}")

    print(f"\nRegistered: {registered}/{len(valid)}")
    if errors:
        print("\nProblems:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("No problems. ✅")
    print("\nVerify with:  python main.py list")
    return registered, errors


def main():
    ap = argparse.ArgumentParser(
        description="Bulk-register stakeholders from an Excel/CSV sheet")
    ap.add_argument("--sheet", required=True,
                    help="Path to .xlsx or .csv (headers: photo, name, role[, uid])")
    ap.add_argument("--photos-dir", default=".",
                    help="Folder containing the photos (default: current dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and preview only — no registration")
    args = ap.parse_args()
    bulk_register(args.sheet, args.photos_dir, args.dry_run)


if __name__ == "__main__":
    main()
