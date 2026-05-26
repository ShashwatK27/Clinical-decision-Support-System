from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr


class ProfileBase(BaseModel):
    organization: Optional[str]
    specialty: Optional[str]
    experience_years: Optional[int]
    preferred_conditions: Optional[str]
    notes: Optional[str]


class ProfileCreate(ProfileBase):
    pass


class ProfileResponse(ProfileBase):
    user_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str]


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    role: str
    created_at: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class PrescriptionDrugBase(BaseModel):
    drug_name: str
    corrected_name: Optional[str]
    dose_value: Optional[str]
    dose_unit: Optional[str]
    frequency: Optional[str]


class PrescriptionCreate(BaseModel):
    input_type: str
    raw_text: str
    ocr_text: Optional[str] = None
    validated_text: Optional[str] = None


class ConditionPredictionResponse(BaseModel):
    condition_label: str
    confidence: Optional[str]
    source: Optional[str]
    evidence: Optional[str]

    class Config:
        orm_mode = True


class PrescriptionResponse(BaseModel):
    id: int
    input_type: str
    raw_text: Optional[str]
    ocr_text: Optional[str]
    validated_text: Optional[str]
    analysis_result: Optional[dict]
    created_at: datetime
    drugs: List[PrescriptionDrugBase] = []
    conditions: List[ConditionPredictionResponse] = []

    class Config:
        orm_mode = True


class ChatMessageBase(BaseModel):
    sender: str
    message_text: str
    msg_metadata: Optional[dict] = None


class ChatMessageResponse(ChatMessageBase):
    created_at: datetime

    class Config:
        orm_mode = True


class ChatSessionResponse(BaseModel):
    id: int
    session_name: Optional[str]
    started_at: datetime
    last_message_at: Optional[datetime]
    context: Optional[dict]
    messages: List[ChatMessageResponse] = []

    class Config:
        orm_mode = True
