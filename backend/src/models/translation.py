from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class Translation(Base):
    __tablename___ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"))
    chapter = relationship("Chapter", back_populates="translations")
    language = Column(String, index=True) # e.g., "urdu"
    title = Column(String)
    content = Column(Text)
