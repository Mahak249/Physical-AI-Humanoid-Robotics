from sqlalchemy.orm import Session
from .. import models, schemas

def get_translation(db: Session, chapter_id: int, language: str):
    return db.query(models.Translation).filter(models.Translation.chapter_id == chapter_id, models.Translation.language == language).first()

def create_chapter_translation(db: Session, translation: schemas.TranslationCreate, chapter_id: int):
    db_translation = models.Translation(**translation.dict(), chapter_id=chapter_id)
    db.add(db_translation)
    db.commit()
    db.refresh(db_translation)
    return db_translation
