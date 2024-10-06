import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.model_services.routers import model_routers

from .config import BaseSettings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
base_config = BaseSettings()
app = FastAPI(
    title="Visionify API",
    version="v1",
    description="This API provides endpoints for visionfy.",
    docs_url="/swagger",
)
app.include_router(model_routers)
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get(
    "/health/",
    summary="Health Check Endpoint",
    description="This endpoint checks the health status of the service.",
)
def health():
    return {"status": "healthy"}
