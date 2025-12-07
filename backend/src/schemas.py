from pydantic import BaseModel
from typing import List, Optional

class TranslationBase(BaseModel):
    language: str
    title: str
    content: str

class TranslationCreate(TranslationBase):
    pass

class Translation(TranslationBase):
    id: int
    chapter_id: int

    class Config:
        orm_mode = True

class ChapterBase(BaseModel):
    title: str
    content: str

class ChapterCreate(ChapterBase):
    pass

class Chapter(ChapterBase):
    id: int
    translations: List[Translation] = []

    class Config:
        orm_mode = True


class UserProfileBase(BaseModel):
    background: str

class UserProfileCreate(UserProfileBase):
    pass

class UserProfile(UserProfileBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
