from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal, Tuple
from uuid import uuid4
from datetime import datetime
import io, csv, json, sqlite3, os

# ======== [新增 import，不會影響原本] ========
from fastapi import UploadFile, File, Form, Header   # 新增：上傳端點需要
from pathlib import Path                              # 新增：處理路徑
import shutil                                         # 新增：寫檔案時複製 file-like 物件

DB_PATH = os.environ.get("FHIR_DB_PATH", "fhir.db")

app = FastAPI(title="Mini FHIR Server (SQLite)", version="0.3.1")

# ---------- CORS（若前端與後端不同來源會需要） ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 可改成你的網域清單
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- FHIR-ish Models (Pydantic v2-friendly) ----------
class HumanName(BaseModel):
    use: Optional[str] = None
    family: Optional[str] = None
    given: Optional[List[str]] = None

class Identifier(BaseModel):
    system: Optional[str] = None
    value: Optional[str] = None

class Patient(BaseModel):
    resourceType: Literal["Patient"] = "Patient"
    id: Optional[str] = None
    identifier: Optional[List[Identifier]] = None
    name: Optional[List[HumanName]] = None
    gender: Optional[Literal["male", "female", "other", "unknown"]] = None
    birthDate: Optional[str] = None  # simplified

class CodeableConcept(BaseModel):
    text: Optional[str] = None

class Reference(BaseModel):
    reference: Optional[str] = None  # e.g., "Patient/123"

class Quantity(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None

class Observation(BaseModel):
    resourceType: Literal["Observation"] = "Observation"
    id: Optional[str] = None
    status: Literal["registered", "preliminary", "final", "amended"]
    code: CodeableConcept
    subject: Reference
    effectiveDateTime: Optional[str] = None
    valueQuantity: Optional[Quantity] = None


# ---------- SQLite helpers ----------
def get_conn():
    # 每個請求新開連線，避免多執行緒共用
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # store full JSON, plus some searchable columns
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        family TEXT,
        given TEXT,
        gender TEXT,
        birthDate TEXT,
        resource_json TEXT NOT NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY,
        subject_ref TEXT,              -- e.g., Patient/{id}
        status TEXT,
        code_text TEXT,
        effectiveDateTime TEXT,
        value_val REAL,
        value_unit TEXT,
        resource_json TEXT NOT NULL
    )""")
    # helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(family, given)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_patients_identifier ON patients(id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_subject ON observations(subject_ref)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_sort ON observations(effectiveDateTime, status, code_text)")
    conn.commit()
    conn.close()

init_db()


# ---------- utilities ----------
def bundle(resources: List[Dict[str, Any]], total: Optional[int] = None):
    """
    FHIR searchset bundle.
    total: 符合條件的總筆數（非本頁筆數）
    """
    if total is None:
        total = len(resources)
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": total,
        "entry": [{"resource": r} for r in resources]
    }

def ensure_id(res: BaseModel):
    if not getattr(res, "id", None):
        setattr(res, "id", str(uuid4()))
    return res.id


# ---------- CapabilityStatement ----------
@app.get("/metadata")
def capability_statement():
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": datetime.utcnow().isoformat(),
        "kind": "instance",
        "software": {"name": "Mini FHIR Server (SQLite)", "version": app.version},
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "rest": [{
            "mode": "server",
            "resource": [
                {"type": "Patient", "interaction": [
                    {"code": "read"}, {"code": "create"}, {"code": "update"}, {"code": "delete"}, {"code": "search-type"}]},
                {"type": "Observation", "interaction": [
                    {"code": "read"}, {"code": "create"}, {"code": "update"}, {"code": "delete"}, {"code": "search-type"}]},
            ]
        }]
    }

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


# ---------- Patient CRUD + Search ----------
@app.post("/Patient", status_code=201)
def create_patient(patient: Patient):
    pid = ensure_id(patient)
    # extract for columns
    n0 = (patient.name or [HumanName()])[0]
    fam = n0.family or ""
    giv = " ".join(n0.given or [])
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients (id, family, given, gender, birthDate, resource_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (pid, fam, giv, patient.gender or None, patient.birthDate or None, json.dumps(patient.model_dump())))
    conn.commit()
    conn.close()
    return patient.model_dump()

@app.get("/Patient/{id_}")
def read_patient(id_: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT resource_json FROM patients WHERE id=?", (id_,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, detail=f"Patient/{id_} not found")
    return json.loads(row["resource_json"])

@app.put("/Patient/{id_}")
def update_patient(id_: str, patient: Patient):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM patients WHERE id=?", (id_,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(404, detail=f"Patient/{id_} not found")
    patient.id = id_
    n0 = (patient.name or [HumanName()])[0]
    fam = n0.family or ""
    giv = " ".join(n0.given or [])
    cur.execute("""
        UPDATE patients SET family=?, given=?, gender=?, birthDate=?, resource_json=?
        WHERE id=?
    """, (fam, giv, patient.gender or None, patient.birthDate or None, json.dumps(patient.model_dump()), id_))
    conn.commit()
    conn.close()
    return patient.model_dump()

@app.delete("/Patient/{id_}", status_code=204)
def delete_patient(id_: str):
    conn = get_conn()
    cur = conn.cursor()
    # delete patient
    cur.execute("DELETE FROM patients WHERE id=?", (id_,))
    # cascade delete observations pointing to it
    cur.execute("DELETE FROM observations WHERE subject_ref=?", (f"Patient/{id_}",))
    conn.commit()
    conn.close()
    return Response(status_code=204)

def _patient_search_sql(
    name: Optional[str],
    identifier: Optional[str],
    _sort: Optional[str],
    _page: int,
    _count: int
) -> Tuple[str, List[Any], str, List[Any]]:
    where = []
    params: List[Any] = []
    if name:
        where.append("(LOWER(family) LIKE ? OR LOWER(given) LIKE ?)")
        like = f"%{name.lower()}%"
        params.extend([like, like])
    if identifier:
        where.append("id = ?")
        params.append(identifier)
    where_sql = " WHERE " + " AND ".join(where) if where else ""

    if _sort in ("name", "-name"):
        order = "DESC" if _sort.startswith("-") else "ASC"
        order_sql = f" ORDER BY family {order}, given {order}"
    else:
        order_sql = " ORDER BY family ASC, given ASC"

    limit = _count
    offset = (_page - 1) * _count

    data_sql = f"SELECT resource_json FROM patients{where_sql}{order_sql} LIMIT ? OFFSET ?"
    data_params = params + [limit, offset]
    count_sql = f"SELECT COUNT(*) AS c FROM patients{where_sql}"
    count_params = list(params)

    return data_sql, data_params, count_sql, count_params

@app.get("/Patient")
def search_patient(
    name: Optional[str] = Query(None),
    identifier: Optional[str] = Query(None),
    _sort: Optional[str] = Query(None, description="name | -name"),
    _page: int = Query(1, ge=1),
    _count: int = Query(50, ge=1, le=200),
):
    data_sql, data_params, count_sql, count_params = _patient_search_sql(name, identifier, _sort, _page, _count)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(data_sql, data_params)
    rows = cur.fetchall()
    cur.execute(count_sql, count_params)
    total = cur.fetchone()["c"]
    conn.close()

    resources = [json.loads(r["resource_json"]) for r in rows]
    return bundle(resources, total=total)


# CSV export (patients)
@app.get("/export/patients.csv")
def export_patients_csv():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, family, given, gender, birthDate FROM patients ORDER BY family, given")
    rows = cur.fetchall()
    conn.close()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "family", "given", "gender", "birthDate"])
    for r in rows:
        writer.writerow([r["id"], r["family"] or "", r["given"] or "", r["gender"] or "", r["birthDate"] or ""])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=patients.csv"}
    )


# ---------- Observation CRUD + Search ----------
@app.post("/Observation", status_code=201)
def create_observation(obs: Observation):
    ref = obs.subject.reference or ""
    if not ref.startswith("Patient/"):
        raise HTTPException(400, detail="subject.reference must be like 'Patient/{id}'")
    pid = ref.split("/", 1)[1]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM patients WHERE id=?", (pid,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(400, detail=f"subject patient not found: {ref}")

    oid = ensure_id(obs)
    code_text = (obs.code.text or "").strip() if obs.code else ""
    val = obs.valueQuantity.value if obs.valueQuantity else None
    unit = obs.valueQuantity.unit if obs.valueQuantity else None
    cur.execute("""
        INSERT INTO observations (id, subject_ref, status, code_text, effectiveDateTime, value_val, value_unit, resource_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (oid, ref, obs.status, code_text or None, obs.effectiveDateTime or None, val, unit, json.dumps(obs.model_dump())))
    conn.commit()
    conn.close()
    return obs.model_dump()

@app.get("/Observation/{id_}")
def read_observation(id_: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT resource_json FROM observations WHERE id=?", (id_,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, detail=f"Observation/{id_} not found")
    return json.loads(row["resource_json"])

@app.put("/Observation/{id_}")
def update_observation(id_: str, obs: Observation):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM observations WHERE id=?", (id_,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(404, detail=f"Observation/{id_} not found")

    ref = obs.subject.reference or ""
    if not ref.startswith("Patient/"):
        conn.close()
        raise HTTPException(400, detail="subject.reference must be like 'Patient/{id}'")
    pid = ref.split("/", 1)[1]
    cur.execute("SELECT 1 FROM patients WHERE id=?", (pid,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(400, detail=f"subject patient not found: {ref}")

    obs.id = id_
    code_text = (obs.code.text or "").strip() if obs.code else ""
    val = obs.valueQuantity.value if obs.valueQuantity else None
    unit = obs.valueQuantity.unit if obs.valueQuantity else None

    cur.execute("""
        UPDATE observations
        SET subject_ref=?, status=?, code_text=?, effectiveDateTime=?, value_val=?, value_unit=?, resource_json=?
        WHERE id=?
    """, (ref, obs.status, code_text or None, obs.effectiveDateTime or None, val, unit, json.dumps(obs.model_dump()), id_))
    conn.commit()
    conn.close()
    return obs.model_dump()

@app.delete("/Observation/{id_}", status_code=204)
def delete_observation(id_: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM observations WHERE id=?", (id_,))
    conn.commit()
    conn.close()
    return Response(status_code=204)

def _obs_search_sql(
    subject: Optional[str],
    _sort: Optional[str],
    _page: int,
    _count: int
) -> Tuple[str, List[Any], str, List[Any]]:
    where = []
    params: List[Any] = []
    if subject:
        where.append("subject_ref = ?")
        params.append(subject)
    where_sql = " WHERE " + " AND ".join(where) if where else ""

    sort_map = {
        "effectiveDateTime": "effectiveDateTime",
        "-effectiveDateTime": "effectiveDateTime DESC",
        "status": "status",
        "-status": "status DESC",
        "code": "code_text",
        "-code": "code_text DESC",
    }
    order_sql = " ORDER BY " + sort_map.get(_sort or "", "effectiveDateTime DESC")

    limit = _count
    offset = (_page - 1) * _count

    data_sql = f"SELECT resource_json FROM observations{where_sql}{order_sql} LIMIT ? OFFSET ?"
    data_params = params + [limit, offset]
    count_sql = f"SELECT COUNT(*) AS c FROM observations{where_sql}"
    count_params = list(params)

    return data_sql, data_params, count_sql, count_params

@app.get("/Observation")
def search_observation(
    subject: Optional[str] = Query(None, description='e.g., "Patient/{id}"'),
    _sort: Optional[str] = Query(None, description="effectiveDateTime | -effectiveDateTime | status | -status | code | -code"),
    _page: int = Query(1, ge=1),
    _count: int = Query(20, ge=1, le=200),
):
    data_sql, data_params, count_sql, count_params = _obs_search_sql(subject, _sort, _page, _count)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(data_sql, data_params)
    rows = cur.fetchall()
    cur.execute(count_sql, count_params)
    total = cur.fetchone()["c"]
    conn.close()

    resources = [json.loads(r["resource_json"]) for r in rows]
    return bundle(resources, total=total)


# ========== [新增] 上傳 / 清單 API ==========
STATIC_DIR = Path(os.getcwd()) / "static"           # 與原本 static_dir 對齊
RESULTS_ROOT = STATIC_DIR / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

UPLOAD_TOKEN = os.environ.get("FHIR_UPLOAD_TOKEN", "change-me")  # 記得在 EC2 設相同環境變數
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".csv", ".xlsx", ".json", ".txt", ".zip"}

def _allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTS

@app.post("/api/upload")
async def api_upload(
    files: List[UploadFile] = File(...),
    run_id: Optional[str] = Form(None),
    x_api_token: Optional[str] = Header(None)
):
    if (x_api_token or "") != UPLOAD_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    if not files:
        raise HTTPException(status_code=400, detail="no files")

    run_id = run_id or datetime.utcnow().strftime("Run_%Y%m%d_%H%M%S")
    dest_dir = RESULTS_ROOT / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_urls: List[str] = []
    for uf in files:
        if not _allowed_file(uf.filename or ""):
            raise HTTPException(status_code=400, detail=f"ext not allowed: {uf.filename}")
        safe_name = Path(uf.filename).name
        dest_path = dest_dir / safe_name
        with dest_path.open("wb") as out:
            shutil.copyfileobj(uf.file, out)
        saved_urls.append(f"/static/results/{run_id}/{safe_name}")

    return {"ok": True, "run_id": run_id, "files": saved_urls}

@app.get("/api/list-runs")
def api_list_runs():
    runs = []
    # 依資料夾名稱排序（你也可以改成 reverse=True 讓最新在最上面）
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        files = []
        # 遞迴抓取允許的副檔名
        for p in run_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                # 轉成可被前端存取的 /static/... 路徑
                rel = p.relative_to(STATIC_DIR)  # STATIC_DIR 是 /opt/FHIR/static
                files.append(f"/static/{rel.as_posix()}")
        runs.append({
            "run_id": run_dir.name,
            "files": files
        })
    return {"ok": True, "runs": runs}


# ---------- UI mounting ----------
@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

# 確保 static 目錄存在（避免部署時未建立導致 404）
static_dir = os.path.join(os.getcwd(), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

# ======== [新增] 讓 /static 路徑也能被存取（提供 /static/results/...） ========
app.mount("/static", StaticFiles(directory=static_dir), name="static")
