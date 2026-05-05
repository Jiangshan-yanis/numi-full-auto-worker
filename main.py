import os
import hmac
import hashlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("numi-auto-worker")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "numi-bronze")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
WORKDIR_ROOT = Path(os.getenv("WORKDIR_ROOT", "/tmp/numi_data_platform"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
app = FastAPI(title="Numi Full Auto Worker")

FOLDER_TO_CATEGORY = {
    "donor_pdf": "donor",
    "protocol_pdf": "protocol",
    "experiment_excel": "experiment",
}

TABLES_BY_CATEGORY = {
    "donor": ["dim_donor_sample"],
    "protocol": ["dim_protocol_entry", "fact_protocol_step"],
    "experiment": ["dim_experiment", "fact_experiment_run", "fact_measurement_long"],
}

# ---------- generic helpers ----------

def verify_secret(received: Optional[str]):
    if not WEBHOOK_SECRET:
        return
    if not received or not hmac.compare_digest(received, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


def ensure_dirs(root: Path):
    for p in [
        root / "bronze" / "donor_pdf",
        root / "bronze" / "protocol_pdf",
        root / "bronze" / "experiment_excel",
        root / "metadata_bronze",
        root / "silver",
        root / "issues",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def list_storage(folder: str):
    items = supabase.storage.from_(BRONZE_BUCKET).list(
        folder,
        {"limit": 1000, "offset": 0, "sortBy": {"column": "name", "order": "asc"}},
    )
    return items or []


def download_folder(folder: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in list_storage(folder):
        name = item.get("name")
        if not name:
            continue
        data = supabase.storage.from_(BRONZE_BUCKET).download(f"{folder}/{name}")
        (local_dir / name).write_bytes(data)


def load_catalog_map(root: Path):
    catalog_path = root / "metadata_bronze" / "bronze_file_catalog.xlsx"
    if not catalog_path.exists():
        return {}
    df = pd.read_excel(catalog_path)
    if "file_name" not in df.columns or "file_id" not in df.columns:
        return {}
    return dict(zip(df["file_name"].astype(str), df["file_id"].astype(str)))


def normalize_text(x: Any) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s or None


def upsert_df(table_name: str, df: pd.DataFrame, on_col: str):
    if df.empty:
        return 0
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    # PostgREST upsert
    supabase.table(table_name).upsert(records, on_conflict=on_col).execute()
    return len(records)


def replace_table(table_name: str, df: pd.DataFrame):
    supabase.table(table_name).delete().neq("id", -1).execute()
    if df.empty:
        return 0
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    batch = 500
    for i in range(0, len(records), batch):
        supabase.table(table_name).insert(records[i:i+batch]).execute()
    return len(records)


def log_sync(status: str, category: str, message: str):
    payload = {"status": status, "message": f"[{category}] {message}"}
    try:
        supabase.table("sync_logs").insert(payload).execute()
    except Exception as e:
        logger.warning("Could not write sync_logs: %s", e)

# ---------- donor parsing ----------

def extract_value(text: str, label: str) -> Optional[str]:
    import re
    m = re.search(rf"{re.escape(label)}\s*:?\s*(.+)", text, flags=re.I)
    return m.group(1).strip() if m else None


def parse_datetime(value: Optional[str]):
    if not value:
        return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def parse_donor_files(root: Path):
    import pdfplumber

    catalog = load_catalog_map(root)
    rows = []
    for pdf_path in sorted((root / "bronze" / "donor_pdf").glob("*.pdf")):
        with pdfplumber.open(str(pdf_path)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        row = {
            "sample_sysid": normalize_text(extract_value(text, "Sysid")),
            "sample_name": normalize_text(text.split("|", 1)[0].strip() if "|" in text else pdf_path.stem),
            "owner_name": normalize_text(extract_value(text, "Owner")),
            "source": normalize_text(extract_value(text, "Source")),
            "created_at": parse_datetime(extract_value(text, "Created at")),
            "collection_datetime": parse_datetime(extract_value(text, "Date and time of collection")),
            "sample_type": normalize_text(extract_value(text, "Type of sample")),
            "number_of_children_raw": normalize_text(extract_value(text, "Number of children")),
            "donor_age_years": pd.to_numeric(extract_value(text, "Age"), errors="coerce"),
            "sample_size_g": normalize_text(extract_value(text, "Sample Size")),
            "serology_raw": normalize_text(extract_value(text, "Serology")),
            "current_medication_raw": normalize_text(extract_value(text, "Current Medication")),
            "pathology_raw": normalize_text(extract_value(text, "Mammary Gland Pathology")),
            "consent_flag": normalize_text(extract_value(text, "Consent")),
            "received_by": normalize_text(extract_value(text, "Received by")),
            "sample_sheet_flag": normalize_text(extract_value(text, "Sample sheet")),
            "transport_sheet_flag": normalize_text(extract_value(text, "Transport sheet")),
            "pickup_datetime": parse_datetime(extract_value(text, "Pick up time by transporter")),
            "dropoff_datetime": parse_datetime(extract_value(text, "Drop off time by transporter") or extract_value(text, "Drop o6 time by transporter") or extract_value(text, "Drop o7 time by transporter")),
            "refrigerated_on_arrival_flag": normalize_text(extract_value(text, "Sample is still refrigerated upon arrival")),
            "origin_of_sample": normalize_text(extract_value(text, "Origin of the sample")),
            "source_file_name": pdf_path.name,
            "source_file_id": catalog.get(pdf_path.name),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- protocol parsing ----------

def parse_protocol_files(root: Path):
    import fitz
    import re

    catalog = load_catalog_map(root)
    dim_rows, fact_rows = [], []
    step_id = 1
    for pdf_path in sorted((root / "bronze" / "protocol_pdf").glob("*.pdf")):
        doc = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        title_line = next((line.strip() for line in text.splitlines() if line.strip()), pdf_path.stem)
        entry_id = None
        m = re.search(r"\bID\s*:\s*(\d+)\b", text)
        if m:
            entry_id = m.group(1)
        else:
            entry_id = pdf_path.stem
        project = normalize_text(extract_value(text, "Project"))
        folder = normalize_text(extract_value(text, "Folder"))
        owner = normalize_text(extract_value(text, "Owner"))
        dim_rows.append({
            "protocol_entry_id": entry_id,
            "title": title_line,
            "project": project,
            "folder": folder,
            "owner_name": owner,
            "source_file_name": pdf_path.name,
            "source_file_id": catalog.get(pdf_path.name),
        })
        # very simple step extraction based on standalone step numbers
        lines = [ln.rstrip() for ln in text.splitlines()]
        current_num = None
        current_buf = []
        def flush_step(num, buf):
            nonlocal step_id
            if num is None:
                return
            content = "\n".join(x for x in buf if x.strip()).strip()
            if not content:
                return
            fact_rows.append({
                "protocol_step_id": f"PSTEP_{step_id:06d}",
                "protocol_entry_id": entry_id,
                "step_number": int(num) if str(num).isdigit() else None,
                "step_title": content.splitlines()[0][:200] if content else None,
                "step_text": content,
                "source_file_name": pdf_path.name,
                "source_file_id": catalog.get(pdf_path.name),
            })
            step_id += 1
        for ln in lines:
            if re.fullmatch(r"\d+", ln.strip()):
                flush_step(current_num, current_buf)
                current_num = ln.strip()
                current_buf = []
            else:
                current_buf.append(ln)
        flush_step(current_num, current_buf)
    return pd.DataFrame(dim_rows), pd.DataFrame(fact_rows)

# ---------- experiment parsing ----------
REQUIRED_MEASUREMENT_MAP = {
    "Viable cell density (10^6 cells/mL)": ("vcd", "10^6 cells/mL"),
    "Viability (%)": ("viability_pct", "%"),
    "Cell diameter (um)": ("cell_diameter_um", "um"),
    "pH": ("ph", "-"),
    "Glucose (mg/L)": ("glucose_mg_L", "mg/L"),
    "Lactate (mg/L)": ("lactate_mg_L", "mg/L"),
    "Ammonia (mg/L)": ("ammonia_mg_L", "mg/L"),
    "LDH (U/L)": ("ldh_u_L", "U/L"),
}
OPTIONAL_MEASUREMENT_MAP = {"Spike Volume (mL)": ("spike_volume_ml", "mL")}


def infer_experiment_metadata(file_name: str) -> dict:
    lower = file_name.lower()
    import re
    digits = re.search(r"exp\s*(\d+)", lower)
    if not digits:
        raise ValueError(f"Cannot infer experiment id from {file_name}")
    exp_num = digits.group(1)
    study_type = "do_strategy" if "do" in lower else "inoculation_density"
    repeat_flag = "repeat" in lower
    default_cell_line_id = "UNKNOWN"
    if "is61" in lower or "s61" in lower:
        default_cell_line_id = "S61"
    if "is70" in lower or "s70" in lower:
        default_cell_line_id = "S70"
    return {
        "experiment_id": f"EXP_{exp_num}",
        "experiment_name": f"Exp {exp_num}",
        "study_type": study_type,
        "repeat_flag": repeat_flag,
        "default_cell_line_id": default_cell_line_id,
    }


def normalize_raw_label(label) -> str:
    if pd.isna(label):
        return ""
    t = str(label).strip()
    if t.startswith("'"):
        t = t[1:].strip()
    return t


def parse_run_label(raw_run_label: str, meta: dict) -> dict:
    import re
    label = normalize_raw_label(raw_run_label)
    if meta["study_type"] == "inoculation_density":
        m = re.match(r"^(\d+)-(\d+(?:\.\d+)?)$", label)
        if not m:
            raise ValueError(f"Cannot parse inoculation run label {raw_run_label}")
        cell_num, cond = m.groups()
        cell_line_id = f"S{cell_num}"
        cond_val = float(cond)
        condition_type = "inoculation_density"
        cond_unit = "10^6 cells/mL"
    else:
        m = re.match(r"^#?(\d+)\s+(\d+(?:\.\d+)?)%\s*DO$", label, flags=re.I)
        if not m:
            raise ValueError(f"Cannot parse DO run label {raw_run_label}")
        _, cond = m.groups()
        cell_line_id = meta["default_cell_line_id"]
        cond_val = float(cond)
        condition_type = "do_strategy"
        cond_unit = "%"
    safe = str(cond_val).replace(".0", "").replace(".", "P")
    run_id = f"RUN_{meta['experiment_id']}_{cell_line_id}_{condition_type.upper()}_{safe}"
    return {
        "run_id": run_id,
        "cell_line_id": cell_line_id,
        "condition_type": condition_type,
        "condition_value_num": cond_val,
        "condition_unit": cond_unit,
        "normalized_raw_run_label": label,
    }


def parse_experiment_files(root: Path):
    catalog = load_catalog_map(root)
    dim_rows, run_rows, meas_rows = [], [], []
    measurement_counter = 1
    for file_path in sorted((root / "bronze" / "experiment_excel").glob("*.xls*")):
        meta = infer_experiment_metadata(file_path.name)
        dim_rows.append({
            "experiment_id": meta["experiment_id"],
            "experiment_name": meta["experiment_name"],
            "study_type": meta["study_type"],
            "cell_line_id": meta["default_cell_line_id"],
            "source_file_name": file_path.name,
            "source_file_id": catalog.get(file_path.name),
        })
        df = pd.read_excel(file_path, engine="openpyxl")
        if "DateID" in df.columns:
            df["DateID"] = pd.to_datetime(df["DateID"], errors="coerce")
        df["ID_normalized"] = df["ID"].apply(normalize_raw_label)
        unique_runs = [x for x in df["ID_normalized"].dropna().astype(str).unique() if x]
        run_lookup = {}
        for raw in sorted(unique_runs):
            parsed = parse_run_label(raw, meta)
            run_rows.append({
                "run_id": parsed["run_id"],
                "experiment_id": meta["experiment_id"],
                "experiment_name": meta["experiment_name"],
                "study_type": meta["study_type"],
                "cell_line_id": parsed["cell_line_id"],
                "raw_run_label": parsed["normalized_raw_run_label"],
                "condition_type": parsed["condition_type"],
                "condition_value_num": parsed["condition_value_num"],
                "condition_unit": parsed["condition_unit"],
                "repeat_flag": meta["repeat_flag"],
                "source_file_name": file_path.name,
                "source_file_id": catalog.get(file_path.name),
            })
            run_lookup[parsed["normalized_raw_run_label"]] = parsed["run_id"]
        for idx, row in df.iterrows():
            raw_label = normalize_raw_label(row.get("ID"))
            if not raw_label:
                continue
            run_id = run_lookup.get(raw_label)
            measurement_date = row["DateID"].date().isoformat() if ("DateID" in df.columns and pd.notna(row["DateID"])) else None
            culture_day = row.get("Day")
            culture_hours = row.get("Hours")
            for raw_col, (std, unit) in REQUIRED_MEASUREMENT_MAP.items():
                value = row.get(raw_col) if raw_col in df.columns else None
                meas_rows.append({
                    "measurement_id": f"M{measurement_counter:06d}",
                    "run_id": run_id,
                    "experiment_id": meta["experiment_id"],
                    "measurement_date": measurement_date,
                    "culture_day": culture_day,
                    "culture_hours": culture_hours,
                    "measure_name_std": std,
                    "measure_name_raw": raw_col,
                    "value_numeric": None if pd.isna(value) else value,
                    "unit": unit,
                    "raw_run_label": raw_label,
                    "source_file_name": file_path.name,
                    "source_file_id": catalog.get(file_path.name),
                    "source_row_number": idx + 2,
                })
                measurement_counter += 1
            for raw_col, (std, unit) in OPTIONAL_MEASUREMENT_MAP.items():
                if raw_col not in df.columns:
                    continue
                value = row.get(raw_col)
                if pd.isna(value):
                    continue
                meas_rows.append({
                    "measurement_id": f"M{measurement_counter:06d}",
                    "run_id": run_id,
                    "experiment_id": meta["experiment_id"],
                    "measurement_date": measurement_date,
                    "culture_day": culture_day,
                    "culture_hours": culture_hours,
                    "measure_name_std": std,
                    "measure_name_raw": raw_col,
                    "value_numeric": value,
                    "unit": unit,
                    "raw_run_label": raw_label,
                    "source_file_name": file_path.name,
                    "source_file_id": catalog.get(file_path.name),
                    "source_row_number": idx + 2,
                })
                measurement_counter += 1
    return pd.DataFrame(dim_rows).drop_duplicates(), pd.DataFrame(run_rows).drop_duplicates(), pd.DataFrame(meas_rows)

# ---------- orchestration ----------

def refresh_category(category: str):
    root = WORKDIR_ROOT
    if root.exists():
        shutil.rmtree(root)
    ensure_dirs(root)

    # metadata catalog optional
    try:
        download_folder("metadata_bronze", root / "metadata_bronze")
    except Exception:
        pass

    if category == "donor":
        download_folder("donor_pdf", root / "bronze" / "donor_pdf")
        df = parse_donor_files(root)
        replace_table("dim_donor_sample", df)
        log_sync("success", category, f"Refreshed dim_donor_sample with {len(df)} rows")
        return {"dim_donor_sample": len(df)}

    if category == "protocol":
        download_folder("protocol_pdf", root / "bronze" / "protocol_pdf")
        dim_df, fact_df = parse_protocol_files(root)
        replace_table("dim_protocol_entry", dim_df)
        replace_table("fact_protocol_step", fact_df)
        log_sync("success", category, f"Refreshed protocol tables: {len(dim_df)} entry rows, {len(fact_df)} step rows")
        return {"dim_protocol_entry": len(dim_df), "fact_protocol_step": len(fact_df)}

    if category == "experiment":
        download_folder("experiment_excel", root / "bronze" / "experiment_excel")
        dim_df, run_df, meas_df = parse_experiment_files(root)
        replace_table("dim_experiment", dim_df)
        replace_table("fact_experiment_run", run_df)
        replace_table("fact_measurement_long", meas_df)
        log_sync("success", category, f"Refreshed experiment tables: {len(dim_df)}, {len(run_df)}, {len(meas_df)} rows")
        return {"dim_experiment": len(dim_df), "fact_experiment_run": len(run_df), "fact_measurement_long": len(meas_df)}

    raise ValueError(f"Unknown category: {category}")


def get_folder_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    record = payload.get("record") or {}
    old_record = payload.get("old_record") or {}
    bucket = (record.get("bucket_id") or old_record.get("bucket_id"))
    if bucket != BRONZE_BUCKET:
        return None
    name = (record.get("name") or old_record.get("name") or "")
    if "/" not in name:
        return None
    folder = name.split("/", 1)[0]
    return folder


@app.get("/")
def health():
    return {"ok": True, "service": "numi-full-auto-worker"}


@app.post("/storage-webhook")
async def storage_webhook(request: Request, x_webhook_secret: Optional[str] = Header(default=None)):
    verify_secret(x_webhook_secret)
    payload = await request.json()
    folder = get_folder_from_payload(payload)
    if not folder or folder not in FOLDER_TO_CATEGORY:
        return {"ok": True, "ignored": True, "reason": "folder not tracked"}
    category = FOLDER_TO_CATEGORY[folder]
    event_type = payload.get("type")
    try:
        result = refresh_category(category)
        return {"ok": True, "category": category, "event": event_type, "result": result}
    except Exception as e:
        logger.exception("Refresh failed")
        log_sync("failed", category, str(e))
        raise HTTPException(status_code=500, detail=str(e))
