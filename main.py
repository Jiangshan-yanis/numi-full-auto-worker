from __future__ import annotations

import os
import hmac
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from supabase import create_client, Client

try:
    import pdfplumber
except ImportError as e:
    raise ImportError("This worker needs pdfplumber. Install with: pip install pdfplumber") from e

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("This worker needs PyMuPDF. Install with: pip install pymupdf") from e


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

OPTIONAL_MEASUREMENT_MAP = {
    "Spike Volume (mL)": ("spike_volume_ml", "mL"),
}

FILE_CELL_LINE_OVERRIDE = {
    "Exp 1680.xlsx": "UNKNOWN",
}


# =========================
# generic helpers
# =========================

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


def clean_value(v: Any):
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def dataframe_to_records(df: pd.DataFrame):
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            rec[col] = clean_value(row[col])
        records.append(rec)
    return records


def replace_table(table_name: str, df: pd.DataFrame):
    if df.empty:
        return 0
    records = dataframe_to_records(df)
    batch = 500
    for i in range(0, len(records), batch):
        supabase.table(table_name).insert(records[i:i + batch]).execute()
    return len(records)


def reset_category_tables(category: str):
    if category == "donor":
        supabase.rpc("reset_donor_tables").execute()
    elif category == "protocol":
        supabase.rpc("reset_protocol_tables").execute()
    elif category == "experiment":
        supabase.rpc("reset_experiment_tables").execute()
    else:
        raise ValueError(f"Unknown category: {category}")


def log_sync(status: str, category: str, message: str):
    payload = {"status": status, "message": f"[{category}] {message}"}
    try:
        supabase.table("sync_logs").insert(payload).execute()
    except Exception as e:
        logger.warning("Could not write sync_logs: %s", e)


def extract_file_id_map(catalog_path: Path) -> Dict[str, str]:
    if not catalog_path.exists():
        return {}
    try:
        df = pd.read_excel(catalog_path)
    except Exception:
        return {}

    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("file_name")
    id_col = cols.get("file_id")
    if not name_col or not id_col:
        return {}

    out = {}
    for _, row in df.iterrows():
        file_name = str(row.get(name_col, "")).strip()
        file_id = str(row.get(id_col, "")).strip()
        if file_name:
            out[file_name] = file_id
    return out


def load_catalog_map(root: Path) -> Dict[str, str]:
    return extract_file_id_map(root / "metadata_bronze" / "bronze_file_catalog.xlsx")


# =========================
# donor logic (aligned with build_silver_donor_table_incremental.py)
# =========================

def collapse_duplicated_letters(text: str) -> str:
    """
    pdfplumber often extracts LabGuru donor PDFs like:
    'SSyyssiidd' -> 'Sysid'
    'OOwwnneerr' -> 'Owner'
    This function removes duplicated adjacent letters only.
    """
    out = []
    prev = None
    for ch in text:
        if ch == prev and ch.isalpha():
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def donor_normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = collapse_duplicated_letters(text)

    replacements = {
        "colection": "collection",
        "Mamary": "Mammary",
        "Curent": "Current",
        "shet": "sheet",
        "stil": "still",
        "arival": "arrival",
        "Atached": "Attached",
        "w.labguru.com": "www.labguru.com",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r":{2,}", ":", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"Rendered by www\.labguru\.com Page \d+ of \d+", "", text, flags=re.I)

    return text.strip()


def extract_donor_pdf_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return donor_normalize_text("\n".join(pages))


def first_match(text: str, patterns: List[str], flags: int = re.I | re.S) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            value = m.group(1).strip()
            return re.sub(r"\s+", " ", value).strip()
    return None


def parse_datetime(value: Optional[str]):
    if not value:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def parse_donor_pdf(pdf_path: Path, file_id_map: Dict[str, str]) -> Tuple[dict, list[dict]]:
    text = extract_donor_pdf_text(pdf_path)
    issues = []

    def find(patterns):
        return first_match(text, patterns)

    sample_name = find([
        r"^([A-Za-z0-9\-_]+)\s*\|\s*patient samples",
    ]) or pdf_path.stem

    row = {
        "sample_sysid": find([r"Sysid:\s*(PS-\d{2}\.\d{4})"]),
        "sample_name": sample_name,
        "owner_name": find([r"Owner:\s*(.+?)\n(?:Source:|Created at:|Date and time of collection:)"]),
        "source_site_raw": find([r"Source:\s*(.+?)\n(?:Created at:|Date and time of collection:)"]) or "Unknown",
        "created_at": parse_datetime(find([r"Created at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"])),
        "collection_datetime": parse_datetime(find([r"Date and time of collection:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})"])),
        "sample_type": find([r"Type of sample:\s*(.+?)\n(?:Number of children:|Age:)"]),
        "sample_size_g": find([r"Sample Size:\s*([0-9.]+\s*g)"]),
        "donor_age_years": pd.to_numeric(find([r"Age:\s*([0-9]+)"]), errors="coerce"),
        "number_of_children_raw": find([r"Number of children:\s*(.+?)\nAge:"]),
        "serology_status": find([r"Serology:\s*(.+?)\n(?:DonorID for MR:|Current Medication:|Consent:)"]),
        "current_medication_raw": find([r"Current Medication:\s*(.+?)\n(?:Mammary Gland Pathology:|Consent:)"]) or "Unknown",
        "pathology_raw": find([r"Mammary Gland Pathology:\s*(.+?)\nConsent:"]) or "Unknown",
        "consent_flag": find([r"Consent:\s*(YES|NO|Yes|No)"]),
        "received_by": find([r"Received by:\s*(.+?)\n(?:Sample sheet:|Transport sheet:)"]),
        "sample_sheet_flag": find([r"Sample sheet:\s*(YES|NO|Yes|No)"]),
        "transport_sheet_flag": find([r"Transport sheet:\s*(YES|NO|Yes|No)"]),
        "pickup_datetime": parse_datetime(find([r"Pick up time by transporter:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})"])),
        "dropoff_datetime": parse_datetime(find([
            r"Drop o\d+\s*time by transporter:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})",
            r"Drop.*?time by transporter:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2})",
        ])),
        "refrigerated_on_arrival_flag": find([
            r"Sample is still refrigerated\s*(YES|NO|Yes|No)\s*upon arrival",
            r"Sample is .*? refrigerated\s*(YES|NO|Yes|No)",
        ]),
        "origin_text": find([r"Origin of the sample:\s*(.+?)\n(?:Attached Images|Linked Resources)"]),
        "source_system": "LabGuru",
        "source_file_name": pdf_path.name,
        "source_file_id": file_id_map.get(pdf_path.name, None),
    }

    required_fields = ["sample_sysid", "sample_name", "created_at"]
    for field in required_fields:
        value = row.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            issues.append({
                "source_file_name": pdf_path.name,
                "issue_type": "missing_required_field",
                "field_name": field,
                "details": "Parser could not extract this field from the PDF text."
            })

    return row, issues


def parse_donor_files(root: Path) -> Tuple[pd.DataFrame, List[dict]]:
    file_id_map = load_catalog_map(root)
    rows = []
    issues = []

    for pdf_path in sorted((root / "bronze" / "donor_pdf").glob("*.pdf")):
        row, row_issues = parse_donor_pdf(pdf_path, file_id_map)
        rows.append(row)
        issues.extend(row_issues)

    return pd.DataFrame(rows), issues


# =========================
# protocol logic (aligned with build_silver_protocol_tables_incremental.py)
# =========================

def extract_protocol_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    text = "\n".join(pages)
    text = text.replace("\r", "\n")

    text = re.sub(r"([0-9]{4}-[0-9]{2}-)\n([0-9]{2})", r"\1\2", text)
    text = re.sub(r"-\n", "-", text)
    text = re.sub(r"PS-\n(\d)", r"PS-\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"Rendered by www\.labguru\.com Page \d+ of \d+", "", text, flags=re.I)
    return text.strip()


def clean_inline_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.replace("•", "; ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(";").strip()
    return text or None


def parse_header_and_sections(text: str, source_file_name: str) -> Tuple[dict, list[dict]]:
    issues = []

    title = first_match(text, [r"^(.+?)\s+Project:"])
    protocol_entry_id = first_match(text, [r"\|\s*ID:\s*(\d+)\s*\|"])
    project_name = first_match(text, [r"Project:\s*(.+?)\s*\|"])
    folder_name = first_match(text, [r"Folder:\s*(.+?)\s*\|"])
    owner_name = first_match(text, [r"Owner:\s*(.+?)\s*\|"])
    created_at_raw = first_match(text, [r"Created at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"])
    background = first_match(text, [r"Background:\s*(.+?)\s*Sources:"])
    expected_results = first_match(text, [r"Expected results:\s*(.+?)\s*Goal"])
    goal_text = first_match(text, [r"Goal\s+(.+?)\s*Sample Description"])
    sample_name = first_match(text, [
        r"Patient samples \(\d+\).*?\n\s*([A-Za-z0-9\-_]+)\s*\n\s*PS-\d{2}\.\d{4}",
        r"Patient samples \(\d+\).*?\n\s*([A-Za-z0-9\-_]+)\s+PS-\d{2}\.\d{4}",
    ])
    sample_sysid = first_match(text, [
        r"Patient samples \(\d+\).*?\n\s*[A-Za-z0-9\-_]+\s*\n\s*(PS-\d{2}\.\d{4})",
        r"Patient samples \(\d+\).*?\n\s*[A-Za-z0-9\-_]+\s+(PS-\d{2}\.\d{4})",
    ])

    row = {
        "protocol_entry_id": pd.to_numeric(protocol_entry_id, errors="coerce"),
        "protocol_title": title,
        "project_name": project_name,
        "folder_name": folder_name,
        "owner_name": owner_name,
        "created_at": pd.to_datetime(created_at_raw, errors="coerce"),
        "goal_text": clean_inline_text(goal_text),
        "background_text": clean_inline_text(background),
        "expected_results_text": clean_inline_text(expected_results),
        "sample_name": sample_name,
        "sample_sysid": sample_sysid,
        "source_file_name": source_file_name,
        "source_system": "LabGuru",
    }

    for field in ["protocol_entry_id", "protocol_title", "project_name"]:
        value = row.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            issues.append({
                "source_file_name": source_file_name,
                "issue_type": "missing_required_field",
                "field_name": field,
                "details": "Parser could not extract this field from the PDF text."
            })

    return row, issues


def is_probable_step_title(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if re.match(r"^[0-9.,%\- ]+$", line):
        return False
    if "|" in line:
        return False
    if line.startswith("Completed by"):
        return False
    if line.endswith(":"):
        return True
    if len(line.split()) >= 3:
        return True
    return False


def should_stop_steps(line: str) -> bool:
    stop_prefixes = [
        "Samples & Reagents",
        "Culture day ",
        "Conclusion",
        "Rendered by ",
    ]
    return any(line.startswith(prefix) for prefix in stop_prefixes)


def normalize_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if re.match(r"^Page \d+ of \d+$", line):
            continue
        if line == "00:00:00":
            continue
        lines.append(line)
    return lines


def extract_step_blocks(lines: List[str]) -> List[List[str]]:
    blocks = []
    current = []
    in_steps = False

    for line in lines:
        if line == "Steps":
            if in_steps and current:
                blocks.append(current)
                current = []
            in_steps = True
            continue

        if in_steps and should_stop_steps(line):
            if current:
                blocks.append(current)
                current = []
            in_steps = False
            continue

        if in_steps:
            current.append(line)

    if in_steps and current:
        blocks.append(current)
    return blocks


def finalize_step(entry_id: int, seq: int, title: str, desc_lines: List[str], completed_by: Optional[str], completed_at_raw: Optional[str]) -> dict:
    title = re.sub(r":\s*$", "", title or "").strip()
    title = re.sub(r"^[•\-]\s*", "", title).strip()
    if not title and desc_lines:
        title = re.sub(r":\s*$", "", desc_lines[0]).strip()
        title = re.sub(r"^[•\-]\s*", "", title).strip()
        desc_lines = desc_lines[1:]

    step_description = " ".join(desc_lines).strip()
    step_description = re.sub(r"\s+", " ", step_description).strip()

    return {
        "protocol_entry_id": entry_id,
        "protocol_step_id": f"P{entry_id}_S{seq:02d}",
        "step_number": seq,
        "step_title": title or f"Step {seq}",
        "step_description": step_description or None,
        "completed_by": completed_by or "unknown",
        "completed_at": pd.to_datetime(completed_at_raw, errors="coerce"),
    }


def parse_steps_from_text(text: str, entry_id: int, source_file_name: str) -> Tuple[List[dict], List[dict]]:
    issues = []
    lines = normalize_lines(text)
    blocks = extract_step_blocks(lines)
    results = []

    seq = 1
    for block in blocks:
        i = 0
        current_title = None
        current_desc = []
        current_completed_by = None
        current_completed_at = None

        def flush_current():
            nonlocal seq, current_title, current_desc, current_completed_by, current_completed_at
            if current_title is None and not current_desc:
                return
            results.append(finalize_step(
                entry_id=entry_id,
                seq=seq,
                title=current_title or "",
                desc_lines=current_desc,
                completed_by=current_completed_by,
                completed_at_raw=current_completed_at,
            ))
            seq += 1
            current_title = None
            current_desc = []
            current_completed_by = None
            current_completed_at = None

        while i < len(block):
            line = block[i]
            next_line = block[i + 1] if i + 1 < len(block) else ""

            m_done = re.search(r"Completed by\s+(.+?)\s+at\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+at\s+\d{2}:\d{2})", line)
            if m_done:
                current_completed_by = m_done.group(1).strip()
                current_completed_at = m_done.group(2).strip()
                i += 1
                continue

            m_inline = re.match(r"^(\d+)\s+(.+)$", line)
            if m_inline and is_probable_step_title(m_inline.group(2)):
                flush_current()
                current_title = m_inline.group(2).strip()
                i += 1
                continue

            if re.match(r"^\d+$", line) and (is_probable_step_title(next_line) or next_line.startswith("•")):
                flush_current()
                if is_probable_step_title(next_line):
                    current_title = next_line.strip()
                    i += 2
                else:
                    current_title = f"Step {seq}"
                    i += 1
                continue

            if re.match(r"^\d{3,}$", line):
                i += 1
                continue

            current_desc.append(line)
            i += 1

        flush_current()

    if not results:
        issues.append({
            "source_file_name": source_file_name,
            "issue_type": "no_steps_parsed",
            "field_name": "fact_protocol_step",
            "details": "No step blocks were parsed from the PDF."
        })

    return results, issues


def parse_protocol_files(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    pdf_files = sorted((root / "bronze" / "protocol_pdf").glob("*.pdf"))
    entry_rows = []
    step_rows = []
    issues = []

    for pdf_path in pdf_files:
        text = extract_protocol_pdf_text(pdf_path)
        entry_row, entry_issues = parse_header_and_sections(text, pdf_path.name)
        entry_rows.append(entry_row)
        issues.extend(entry_issues)

        entry_id = entry_row.get("protocol_entry_id")
        if pd.isna(entry_id):
            issues.append({
                "source_file_name": pdf_path.name,
                "issue_type": "missing_protocol_entry_id",
                "field_name": "protocol_entry_id",
                "details": "Cannot parse steps without a protocol_entry_id."
            })
            continue

        parsed_steps, step_issues = parse_steps_from_text(text, int(entry_id), pdf_path.name)
        step_rows.extend(parsed_steps)
        issues.extend(step_issues)

    return pd.DataFrame(entry_rows), pd.DataFrame(step_rows), issues


# =========================
# experiment logic (aligned with build_silver_experiment_tables_incremental.py)
# =========================

def normalize_raw_label(label) -> str:
    if pd.isna(label):
        return ""
    text = str(label).strip()
    if text.startswith("'"):
        text = text[1:].strip()
    return re.sub(r"\s+", " ", text).strip()


def is_missing(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def format_value_for_id(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "P").replace("-", "NEG")


def slugify(text: str) -> str:
    text = normalize_raw_label(text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_") or "UNKNOWN"


def parse_bool_from_filename(file_name: str) -> bool:
    return bool(re.search(r"\brepeat\b|\breplicate\b|\brep\b", file_name, flags=re.I))


def infer_cell_line_from_filename(file_name: str) -> str:
    override = FILE_CELL_LINE_OVERRIDE.get(file_name)
    if override:
        return override

    m = re.search(r"i?S(\d+)", file_name, flags=re.I)
    if m:
        return f"S{m.group(1)}"

    return "UNKNOWN"


def infer_study_type(file_name: str) -> str:
    name = file_name.lower()
    if "inoculation density" in name:
        return "inoculation_density"
    if "do strategy" in name:
        return "do_strategy"
    if "dissolved oxygen" in name or re.search(r"\bdo\b", name):
        return "do_strategy"
    return "unknown"


def infer_experiment_id(file_name: str) -> str:
    m = re.search(r"\bexp\D*?(\d+)\b", file_name, flags=re.I)
    if m:
        return f"EXP_{m.group(1)}"
    return f"EXP_FILE_{slugify(Path(file_name).stem)}"


def infer_experiment_name(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def infer_experiment_metadata(file_name: str) -> dict:
    return {
        "experiment_id": infer_experiment_id(file_name),
        "experiment_name": infer_experiment_name(file_name),
        "study_type": infer_study_type(file_name),
        "repeat_flag": parse_bool_from_filename(file_name),
        "default_cell_line_id": infer_cell_line_from_filename(file_name),
    }


def parse_run_label(raw_run_label: str, meta: dict, file_name: str) -> Tuple[dict, list[dict]]:
    issues = []
    label = normalize_raw_label(raw_run_label)

    m = re.match(r"^(?:S)?(\d+)-(\d+(?:\.\d+)?)$", label, flags=re.I)
    if m:
        cell_num, cond = m.groups()
        cell_line_id = f"S{cell_num}"
        cond_val = float(cond)
        condition_type = "inoculation_density"
        cond_unit = "10^6 cells/mL"
        run_id = f"RUN_{meta['experiment_id']}_{cell_line_id}_{condition_type.upper()}_{format_value_for_id(cond_val)}"
        return ({
            "run_id": run_id,
            "cell_line_id": cell_line_id,
            "condition_type": condition_type,
            "condition_value_num": cond_val,
            "condition_unit": cond_unit,
            "normalized_raw_run_label": label,
        }, issues)

    m = re.match(r"^#?(\d+)\s+(\d+(?:\.\d+)?)\s*%\s*DO$", label, flags=re.I)
    if m:
        replicate_num, cond = m.groups()
        cell_line_id = infer_cell_line_from_filename(file_name)
        cond_val = float(cond)
        condition_type = "do_strategy"
        cond_unit = "%"
        run_id = f"RUN_{meta['experiment_id']}_{cell_line_id}_{condition_type.upper()}_{format_value_for_id(cond_val)}_R{replicate_num}"
        return ({
            "run_id": run_id,
            "cell_line_id": cell_line_id,
            "condition_type": condition_type,
            "condition_value_num": cond_val,
            "condition_unit": cond_unit,
            "normalized_raw_run_label": label,
        }, issues)

    issues.append({
        "source_file_name": file_name,
        "issue_type": "unparsed_run_label",
        "field_name": "raw_run_label",
        "details": f"Could not parse run label '{label}'. Fallback generic run_id was used."
    })
    cell_line_id = infer_cell_line_from_filename(file_name)
    run_id = f"RUN_{meta['experiment_id']}_{cell_line_id}_{slugify(label)}"
    return ({
        "run_id": run_id,
        "cell_line_id": cell_line_id,
        "condition_type": meta["study_type"],
        "condition_value_num": None,
        "condition_unit": None,
        "normalized_raw_run_label": label,
    }, issues)


def pick_best_sheet(xls: pd.ExcelFile) -> str:
    preferred = [s for s in xls.sheet_names if s.lower() in {"raw data", "sheet1"}]
    if preferred:
        return preferred[0]
    return xls.sheet_names[0]


def parse_single_experiment_file(file_path: Path, file_id_map: Dict[str, str], measurement_counter_start: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], int]:
    issues = []
    meta = infer_experiment_metadata(file_path.name)
    file_id = file_id_map.get(file_path.name)

    xls = pd.ExcelFile(file_path)
    sheet_name = pick_best_sheet(xls)
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    required_base_cols = ["ID", "DateID", "Day", "Hours"]
    for col in required_base_cols:
        if col not in df.columns:
            issues.append({
                "source_file_name": file_path.name,
                "issue_type": "missing_required_column",
                "field_name": col,
                "details": "The file is missing a required base column."
            })

    if "ID" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), issues, measurement_counter_start

    if "DateID" in df.columns:
        df["DateID"] = pd.to_datetime(df["DateID"], errors="coerce")

    df["ID_normalized"] = df["ID"].apply(normalize_raw_label)

    if meta["study_type"] == "unknown":
        labels = " || ".join(df["ID_normalized"].dropna().astype(str).tolist())
        if re.search(r"%\s*DO", labels, flags=re.I):
            meta["study_type"] = "do_strategy"
        elif re.search(r"(?:^|\|\|)\s*(?:S)?\d+-\d+(?:\.\d+)?", labels, flags=re.I):
            meta["study_type"] = "inoculation_density"

    unique_cell_lines = []
    inferred_from_file = infer_cell_line_from_filename(file_path.name)
    if inferred_from_file != "UNKNOWN":
        unique_cell_lines.append(inferred_from_file)
    for raw_run in df["ID_normalized"].dropna().astype(str).unique():
        m = re.match(r"^(?:S)?(\d+)-", raw_run, flags=re.I)
        if m:
            unique_cell_lines.append(f"S{m.group(1)}")
    unique_cell_lines = sorted(set([x for x in unique_cell_lines if x]))
    experiment_cell_line = unique_cell_lines[0] if len(unique_cell_lines) == 1 else (unique_cell_lines[0] if unique_cell_lines else "UNKNOWN")

    dim_experiment_row = pd.DataFrame([{
        "experiment_id": meta["experiment_id"],
        "experiment_name": meta["experiment_name"],
        "study_type": meta["study_type"],
        "cell_line_id": experiment_cell_line,
        "source_file_name": file_path.name,
        "source_file_id": file_id,
        "file_format": file_path.suffix.lower().lstrip("."),
    }])

    run_records = []
    run_lookup = {}
    for raw_run_label in sorted([x for x in df["ID_normalized"].dropna().astype(str).unique() if x != ""]):
        parsed, parse_issues = parse_run_label(raw_run_label, meta, file_path.name)
        issues.extend(parse_issues)

        run_row = {
            "run_id": parsed["run_id"],
            "experiment_id": meta["experiment_id"],
            "study_type": meta["study_type"],
            "cell_line_id": parsed["cell_line_id"],
            "raw_run_label": parsed["normalized_raw_run_label"],
            "condition_type": parsed["condition_type"],
            "condition_value_num": parsed["condition_value_num"],
            "condition_unit": parsed["condition_unit"],
            "repeat_flag": meta["repeat_flag"],
            "source_file_name": file_path.name,
            "source_file_id": file_id,
            "source_sheet_name": sheet_name,
            "source_system": "manual_export",
            "file_format": file_path.suffix.lower().lstrip("."),
        }
        run_records.append(run_row)
        run_lookup[parsed["normalized_raw_run_label"]] = parsed["run_id"]

    run_df = pd.DataFrame(run_records)

    measurement_records = []
    measurement_counter = measurement_counter_start

    for _, row in df.iterrows():
        raw_run_label = normalize_raw_label(row.get("ID"))
        if raw_run_label == "":
            continue

        run_id = run_lookup.get(raw_run_label)
        measurement_date = (
            row["DateID"].date().isoformat()
            if ("DateID" in df.columns and pd.notna(row["DateID"]))
            else None
        )
        culture_day = row.get("Day")
        culture_hours = row.get("Hours")

        for raw_col, (measure_name_std, unit) in REQUIRED_MEASUREMENT_MAP.items():
            value = row.get(raw_col) if raw_col in df.columns else None
            value_out = "UNKNOWN" if is_missing(value) else value

            measurement_records.append({
                "measurement_id": f"M{measurement_counter:06d}",
                "run_id": run_id,
                "experiment_id": meta["experiment_id"],
                "measurement_date": measurement_date,
                "culture_day": culture_day,
                "culture_hours": culture_hours,
                "measure_name_std": measure_name_std,
                "measure_name_raw": raw_col,
                "value_numeric": value_out,
                "unit": unit,
                "raw_run_label": raw_run_label,
                "source_file_name": file_path.name,
                "source_sheet_name": sheet_name,
                "source_system": "manual_export",
            })
            measurement_counter += 1

        for raw_col, (measure_name_std, unit) in OPTIONAL_MEASUREMENT_MAP.items():
            if raw_col not in df.columns:
                continue

            value = row.get(raw_col)
            if is_missing(value):
                continue

            measurement_records.append({
                "measurement_id": f"M{measurement_counter:06d}",
                "run_id": run_id,
                "experiment_id": meta["experiment_id"],
                "measurement_date": measurement_date,
                "culture_day": culture_day,
                "culture_hours": culture_hours,
                "measure_name_std": measure_name_std,
                "measure_name_raw": raw_col,
                "value_numeric": value,
                "unit": unit,
                "raw_run_label": raw_run_label,
                "source_file_name": file_path.name,
                "source_sheet_name": sheet_name,
                "source_system": "manual_export",
            })
            measurement_counter += 1

    measurement_df = pd.DataFrame(measurement_records)
    return dim_experiment_row, run_df, measurement_df, issues, measurement_counter


def parse_experiment_files(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    bronze_dir = root / "bronze" / "experiment_excel"
    file_id_map = load_catalog_map(root)
    excel_files = sorted([p for p in bronze_dir.iterdir() if p.suffix.lower() in [".xlsx", ".xls"]])
    if not excel_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    measurement_counter = 1
    new_dim_parts = []
    new_run_parts = []
    new_meas_parts = []
    issues = []

    for file_path in excel_files:
        dim_part, run_part, meas_part, part_issues, measurement_counter = parse_single_experiment_file(
            file_path=file_path,
            file_id_map=file_id_map,
            measurement_counter_start=measurement_counter,
        )
        if not dim_part.empty:
            new_dim_parts.append(dim_part)
        if not run_part.empty:
            new_run_parts.append(run_part)
        if not meas_part.empty:
            new_meas_parts.append(meas_part)
        issues.extend(part_issues)

    new_dim_df = pd.concat(new_dim_parts, ignore_index=True) if new_dim_parts else pd.DataFrame()
    new_run_df = pd.concat(new_run_parts, ignore_index=True) if new_run_parts else pd.DataFrame()
    new_meas_df = pd.concat(new_meas_parts, ignore_index=True) if new_meas_parts else pd.DataFrame()

    if not new_dim_df.empty:
        new_dim_df = new_dim_df.drop_duplicates(subset=["experiment_id"], keep="last")
        new_dim_df = new_dim_df.sort_values(["experiment_id"], kind="stable").reset_index(drop=True)

    if not new_run_df.empty:
        new_run_df = new_run_df.drop_duplicates(subset=["run_id"], keep="last")
        new_run_df = new_run_df.sort_values(["experiment_id", "condition_value_num", "raw_run_label"], kind="stable").reset_index(drop=True)

    if not new_meas_df.empty:
        dedupe_cols = [
            "run_id", "experiment_id", "measurement_date", "culture_day", "culture_hours",
            "measure_name_std", "raw_run_label", "source_file_name"
        ]
        dedupe_cols = [c for c in dedupe_cols if c in new_meas_df.columns]
        new_meas_df = new_meas_df.drop_duplicates(subset=dedupe_cols, keep="last")
        new_meas_df = new_meas_df.sort_values(
            ["experiment_id", "run_id", "culture_day", "culture_hours", "measure_name_std"],
            kind="stable"
        ).reset_index(drop=True)
        new_meas_df["measurement_id"] = [f"M{i:06d}" for i in range(1, len(new_meas_df) + 1)]

    return new_dim_df, new_run_df, new_meas_df, issues


# =========================
# orchestration
# =========================

def refresh_category(category: str):
    root = WORKDIR_ROOT
    if root.exists():
        shutil.rmtree(root)
    ensure_dirs(root)

    try:
        download_folder("metadata_bronze", root / "metadata_bronze")
    except Exception:
        pass

    reset_category_tables(category)

    if category == "donor":
        download_folder("donor_pdf", root / "bronze" / "donor_pdf")
        donor_df, issues = parse_donor_files(root)
        count = replace_table("dim_donor_sample", donor_df)
        if issues:
            logger.warning("Donor parse issues: %s", issues)
        log_sync("success", category, f"Refreshed dim_donor_sample with {count} rows")
        return {"dim_donor_sample": count}

    if category == "protocol":
        download_folder("protocol_pdf", root / "bronze" / "protocol_pdf")
        entry_df, step_df, issues = parse_protocol_files(root)
        count_entry = replace_table("dim_protocol_entry", entry_df)
        count_step = replace_table("fact_protocol_step", step_df)
        if issues:
            logger.warning("Protocol parse issues: %s", issues)
        log_sync("success", category, f"Refreshed protocol tables: {count_entry} entry rows, {count_step} step rows")
        return {"dim_protocol_entry": count_entry, "fact_protocol_step": count_step}

    if category == "experiment":
        download_folder("experiment_excel", root / "bronze" / "experiment_excel")
        dim_df, run_df, meas_df, issues = parse_experiment_files(root)
        count_dim = replace_table("dim_experiment", dim_df)
        count_run = replace_table("fact_experiment_run", run_df)
        count_meas = replace_table("fact_measurement_long", meas_df)
        if issues:
            logger.warning("Experiment parse issues: %s", issues)
        log_sync("success", category, f"Refreshed experiment tables: {count_dim}, {count_run}, {count_meas} rows")
        return {
            "dim_experiment": count_dim,
            "fact_experiment_run": count_run,
            "fact_measurement_long": count_meas,
        }

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
