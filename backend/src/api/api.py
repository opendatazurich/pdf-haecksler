from fastapi import APIRouter
from . import auth, requests

router = APIRouter()
router.include_router(auth.router, tags=["authentication"], prefix="/users")
router.include_router(requests.router, tags=["functions"])
