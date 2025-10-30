import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables to get the Redis URL
load_dotenv()

redis_url = os.getenv("REDIS_URL")

if not redis_url:
    raise ValueError("REDIS_URL environment variable not set.")

celery_app = Celery(
    "api_client",
    broker=redis_url,
    backend=redis_url 
) 