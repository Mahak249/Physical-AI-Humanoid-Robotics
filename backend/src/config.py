from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    qdrant_url: str
    qdrant_api_key: str
    openai_api_key: str
    better_auth_secret_key: str

    class Config:
        env_file = ".env"

settings = Settings()
