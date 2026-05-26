import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from . import models, crud, schemas, auth
from .database import engine, get_db
from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from embeddings.embedding import get_embedding
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from vector_db.store import VectorStore

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None  # type: ignore
    pytesseract = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CDSS API",
    description="Clinical Decision Support backend for prescription analysis, OCR, auth, and history.",
    version="0.1.0",
)

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

store: Optional[VectorStore] = None
mapper: Optional[ConditionMapper] = None


def init_runtime() -> None:
    global store, mapper
    if store is not None and mapper is not None:
        return

    mapper = ConditionMapper()
    store = VectorStore.load(str(BASE_DIR / "vector_store"))
    if store is None:
        store = VectorStore()
        fallback = [
            ("ibuprofen pain", {"drugs": ["ibuprofen"], "conditions": ["pain"]}),
            ("metformin diabetes", {"drugs": ["metformin"], "conditions": ["diabetes"]}),
            ("paracetamol fever", {"drugs": ["paracetamol"], "conditions": ["fever"]}),
        ]
        for text, metadata in fallback:
            store.add(get_embedding(text), metadata)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    payload = auth.decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = crud.get_user_by_email(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def analyze_prescription(raw_text: str) -> Dict[str, Any]:
    init_runtime()
    parsed = parse_prescription(raw_text)
    corrected_drugs = correct_drug_list(parsed["drugs"])
    cleaned = clean_medications(parsed["medications"])
    query_vector = get_embedding(build_embedding_text(corrected_drugs, cleaned))
    search_results = store.search(query_vector, top_k=5, threshold=0.30)

    predictions = mapper.predict(corrected_drugs, vector_results=search_results)
    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "corrected_drugs": corrected_drugs,
        "search_results": [
            {"metadata": meta, "score": score} for meta, score in search_results
        ],
        "conditions": predictions,
    }


@app.post("/api/register", response_model=schemas.UserResponse)
def register_user(user_create: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, user_create.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = crud.create_user(db, user_create)
    return user


@app.post("/api/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token({"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/me", response_model=schemas.UserResponse)
def read_current_user(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.post("/api/prescriptions/", response_model=schemas.PrescriptionResponse)
def create_prescription_endpoint(
    payload: schemas.PrescriptionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = analyze_prescription(payload.raw_text)
    prescription = crud.create_prescription(db, current_user.id, payload, analysis)
    crud.create_condition_predictions(db, prescription.id, analysis.get("conditions", []))
    db.refresh(prescription)
    return prescription


@app.get("/api/prescriptions/", response_model=List[schemas.PrescriptionResponse])
def list_prescriptions(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return crud.list_prescriptions(db, current_user.id)


@app.post("/api/ocr/")
async def extract_text_from_image(file: UploadFile = File(...)):
    if Image is None or pytesseract is None:
        raise HTTPException(
            status_code=500,
            detail="OCR dependencies are not installed. Install Pillow and pytesseract to enable scanned prescription support.",
        )

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    text = pytesseract.image_to_string(image)
    return {"ocr_text": text}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
