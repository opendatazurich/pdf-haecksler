from fastapi import FastAPI, Depends, HTTPException, APIRouter
from sqlalchemy.orm import Session
from .auth_handler import AuthHandler
from .schemas import AuthDetails
from .database import get_db
from .models import User

auth_handler = AuthHandler()
router = APIRouter()

@router.post("/register",status_code=201,name="auth:register")
def register(auth_details: AuthDetails, db: Session = Depends(get_db)):
    usernames = [*map(lambda u: u.username, db.query(User).all())]
    if auth_details.username in usernames:
        raise HTTPException(status_code=400, detail='Username is taken')
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    to_create = User(
        username=auth_details.username,
        password=hashed_password
    )
    db.add(to_create)
    db.commit()
    return {"success": True}

@router.post("/login", name="auth:login")
def login(auth_details: AuthDetails, db: Session = Depends(get_db)):
    user = None
    users = [*db.query(User).all()]
    for x in users:
        if x.username == auth_details.username:
            user = x
            break
    if (user is None) or (not auth_handler.verify_password(auth_details.password, user.password)):
        raise HTTPException(status_code=401, detail='Invalid username and/or password')
    token = auth_handler.encode_token(user.username)
    return {'token': token}
