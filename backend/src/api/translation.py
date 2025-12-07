from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import schemas
from ..services import translation as translation_service
from ..services import chapter as chapter_service
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/chapters/{chapter_id}/translations", response_model=schemas.Translation)
def create_translation_for_chapter(
    chapter_id: int, translation: schemas.TranslationCreate, db: Session = Depends(get_db)
):
    return translation_service.create_chapter_translation(db=db, translation=translation, chapter_id=chapter_id)

@router.get("/chapters/{chapter_id}/translations/{language}", response_model=schemas.Translation)
def read_translation(chapter_id: int, language: str, db: Session = Depends(get_db)):
    db_translation = translation_service.get_translation(db, chapter_id=chapter_id, language=language)
    if db_translation is None:
        raise HTTPException(status_code=404, detail="Translation not found")
    return db_translation
