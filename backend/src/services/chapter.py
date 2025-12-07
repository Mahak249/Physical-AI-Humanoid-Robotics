from sqlalchemy.orm import Session
from .. import models, schemas

def get_chapter(db: Session, chapter_id: int):
    return db.query(models.Chapter).filter(models.Chapter.id == chapter_id).first()

def create_chapter(db: Session, chapter: schemas.ChapterCreate):
    db_chapter = models.Chapter(**chapter.dict())
    db.add(db_chapter)
    db.commit()
    db.refresh(db_chapter)
    return db_chapter
