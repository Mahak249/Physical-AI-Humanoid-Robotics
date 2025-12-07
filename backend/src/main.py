from fastapi import FastAPI
from .api.main import router as api_router
from .api.user_profile import router as user_profile_router
from .api.chapter import router as chapter_router
from .api.translation import router as translation_router

app = FastAPI()

app.include_router(api_router, prefix="/api")
app.include_router(user_profile_router, prefix="/api")
app.include_router(chapter_router, prefix="/api")
app.include_router(translation_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Server is running"}
