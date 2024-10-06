from fastapi import APIRouter

from .views import router

model_routers = APIRouter()

model_routers.include_router(router, prefix="/models", tags=["models"])
