from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import schemas
from ..services import user_profile as user_profile_service
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/users/{user_id}/profile", response_model=schemas.UserProfile)
def create_user_profile(user_id: int, profile: schemas.UserProfileCreate, db: Session = Depends(get_db)):
    return user_profile_service.create_user_profile(db=db, profile=profile, user_id=user_id)

@router.get("/users/{user_id}/profile", response_model=schemas.UserProfile)
def read_user_profile(user_id: int, db: Session = Depends(get_db)):
    db_user_profile = user_profile_service.get_user_profile(db, user_id=user_id)
    if db_user_profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return db_user_profile
