from datetime import datetime
from sqlalchemy.orm import Session
from typing import List, Optional

from . import models, schemas
from .auth import get_password_hash, verify_password


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()


def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    user_obj = models.User(
        email=user.email,
        password_hash=get_password_hash(user.password),
        full_name=user.full_name,
    )
    db.add(user_obj)
    db.commit()
    db.refresh(user_obj)
    return user_obj


def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    user.last_login = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_prescriptions(db: Session, user_id: int) -> List[models.Prescription]:
    return db.query(models.Prescription).filter(models.Prescription.user_id == user_id).order_by(models.Prescription.created_at.desc()).all()


def create_prescription(db: Session, user_id: int, data: schemas.PrescriptionCreate, analysis_result: dict) -> models.Prescription:
    prescription_obj = models.Prescription(
        user_id=user_id,
        input_type=data.input_type,
        raw_text=data.raw_text,
        ocr_text=data.ocr_text,
        validated_text=data.validated_text,
        analysis_result=analysis_result,
    )
    db.add(prescription_obj)
    db.commit()
    db.refresh(prescription_obj)
    return prescription_obj


def create_condition_predictions(db: Session, prescription_id: int, predictions: List[dict]) -> List[models.ConditionPrediction]:
    result = []
    for prediction in predictions:
        prediction_obj = models.ConditionPrediction(
            prescription_id=prescription_id,
            condition_label=prediction.get("condition_label", "unknown"),
            confidence=str(prediction.get("confidence", "")),
            source=prediction.get("source", "hybrid"),
            evidence=prediction.get("evidence", ""),
        )
        db.add(prediction_obj)
        result.append(prediction_obj)
    db.commit()
    return result
