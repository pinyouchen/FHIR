from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal
from uuid import uuid4
from datetime import datetime
import io, csv, json, sqlite3, os

DB_PATH = os.environ.get("FHIR_DB_PATH", "fhir.db")

app = FastAPI(title="Mini FHIR Server (SQLite)", version="0.3.0")

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
def bundle(resources: List[Dict[str, Any]]):
    return {"resourceType": "Bundle", "type": "searchset", "total": len(resources),
            "entry": [{"resource": r} for r in resources]}

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
        "software": {"name": "Mini FHIR Server (SQLite)", "version": "0.3.0"},
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
    return ""

@app.get("/Patient")
def search_patient(
    name: Optional[str] = Query(None),
    identifier: Optional[str] = Query(None),
    _sort: Optional[str] = Query(None, description="name | -name"),
    _page: int = Query(1, ge=1),
    _count: int = Query(50, ge=1, le=200),
):
    where = []
    params: List[Any] = []
    if name:
        # match family or given contains (case-insensitive)
        where.append("(LOWER(family) LIKE ? OR LOWER(given) LIKE ?)")
        like = f"%{name.lower()}%"
        params.extend([like, like])

    if identifier:
        # identifier here就是 id（我們用 id 當主鍵）
        where.append("id = ?")
        params.append(identifier)

    where_sql = " WHERE " + " AND ".join(where) if where else ""
    # sort
    if _sort in ("name", "-name"):
        order = "DESC" if _sort.startswith("-") else "ASC"
        order_sql = f" ORDER BY family {order}, given {order}"
    else:
        order_sql = " ORDER BY family ASC, given ASC"

    # pagination
    limit = _count
    offset = (_page - 1) * _count

    sql = f"SELECT resource_json FROM patients{where_sql}{order_sql} LIMIT ? OFFSET ?"
    params_with_page = params + [limit, offset]

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params_with_page)
    rows = cur.fetchall()
    conn.close()

    resources = [json.loads(r["resource_json"]) for r in rows]
    return bundle(resources)


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
    return StreamingResponse(iter([buf.read()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=patients.csv"})


# ---------- Observation CRUD + Search ----------
@app.post("/Observation", status_code=201)
def create_observation(obs: Observation):
    ref = obs.subject.reference or ""
    if not ref.startswith("Patient/"):
        raise HTTPException(400, detail="subject.reference must be like 'Patient/{id}'")
    pid = ref.split("/", 1)[1]
    # patient exists?
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
    return ""

@app.get("/Observation")
def search_observation(
    subject: Optional[str] = Query(None),  # "Patient/{id}"
    _sort: Optional[str] = Query(None, description="effectiveDateTime | -effectiveDateTime | status | -status | code | -code"),
    _page: int = Query(1, ge=1),
    _count: int = Query(20, ge=1, le=200),
):
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

    sql = f"SELECT resource_json FROM observations{where_sql}{order_sql} LIMIT ? OFFSET ?"
    params_with_page = params + [limit, offset]

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params_with_page)
    rows = cur.fetchall()
    conn.close()

    resources = [json.loads(r["resource_json"]) for r in rows]
    return bundle(resources)


# ---------- UI mounting ----------
@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
