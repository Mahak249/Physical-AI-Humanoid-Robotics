from sqlalchemy.orm import Session
from .. import models, schemas

def get_user_profile(db: Session, user_id: int):
    return db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()

def create_user_profile(db: Session, profile: schemas.UserProfileCreate, user_id: int):
    db_profile = models.UserProfile(**profile.dict(), user_id=user_id)
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile
