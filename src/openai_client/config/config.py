import os
from dotenv import load_dotenv
from pydantic import BaseModel

class Config(BaseModel):
    api_key: str
    base_url: str = "http://localhost:8000/v1"
    organization_id: str = "org-123"

def load() -> Config:
    """Load configuration from environment variables."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    organization_id = os.getenv("OPENAI_ORGANIZATION_ID", "org-123")
    
    return Config(
        api_key=api_key,
        base_url=base_url,
        organization_id=organization_id
    ) 