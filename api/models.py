from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="clinician")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    profile = relationship("Profile", back_populates="user", uselist=False)
    prescriptions = relationship("Prescription", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization = Column(String, nullable=True)
    specialty = Column(String, nullable=True)
    experience_years = Column(Integer, nullable=True)
    preferred_conditions = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="profile")


class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    input_type = Column(String, default="text")
    raw_text = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    validated_text = Column(Text, nullable=True)
    analysis_result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="prescriptions")
    drugs = relationship("PrescriptionDrug", back_populates="prescription", cascade="all, delete")
    conditions = relationship("ConditionPrediction", back_populates="prescription", cascade="all, delete")


class PrescriptionDrug(Base):
    __tablename__ = "prescription_drugs"

    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id", ondelete="CASCADE"), nullable=False)
    drug_name = Column(String, nullable=False)
    corrected_name = Column(String, nullable=True)
    dose_value = Column(String, nullable=True)
    dose_unit = Column(String, nullable=True)
    frequency = Column(String, nullable=True)

    prescription = relationship("Prescription", back_populates="drugs")


class ConditionPrediction(Base):
    __tablename__ = "condition_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prescription_id = Column(Integer, ForeignKey("prescriptions.id", ondelete="CASCADE"), nullable=False)
    condition_label = Column(String, nullable=False)
    confidence = Column(String, nullable=True)
    source = Column(String, nullable=True)
    evidence = Column(Text, nullable=True)

    prescription = relationship("Prescription", back_populates="conditions")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_name = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, nullable=True)
    context = Column(JSON, nullable=True)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    sender = Column(String, nullable=False)
    message_text = Column(Text, nullable=False)
    msg_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")
