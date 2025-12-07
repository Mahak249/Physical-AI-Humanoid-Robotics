from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import schemas
from ..services import chapter as chapter_service
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/chapters", response_model=schemas.Chapter)
def create_chapter(chapter: schemas.ChapterCreate, db: Session = Depends(get_db)):
    return chapter_service.create_chapter(db=db, chapter=chapter)

@router.get("/chapters/{chapter_id}", response_model=schemas.Chapter)
def read_chapter(chapter_id: int, db: Session = Depends(get_db)):
    db_chapter = chapter_service.get_chapter(db, chapter_id=chapter_id)
    return db_chapter
