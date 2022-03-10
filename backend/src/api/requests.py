from fastapi import FastAPI, Depends, HTTPException, APIRouter
from .auth_handler import AuthHandler
# from src.app.main import *

auth_handler = AuthHandler()
router = APIRouter()

@router.get("/unprotected",status_code=201,name="test:unprotected")
def unprotected():
    #x = check("hello","world")
    #return x
    return "hello world"

@router.get("/protected",status_code=201,name="test:protected")
def protected(username=Depends(auth_handler.auth_wrapper)):
    return { 'name': username }
