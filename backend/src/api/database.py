from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import String, Integer, Float, Boolean, Column
from sqlalchemy import create_engine, Sequence
import os

DB_PWD = os.getenv("POSTGRES_PASSWORD")
DB_USR = os.getenv("POSTGRES_USER")
DB_SNM = os.getenv("POSTGRES_SERVICE_NAME")
DB_NAM = os.getenv("POSTGRES_DB")
DB_URI = f"postgresql://{DB_USR}:{DB_PWD}@{DB_SNM}/{DB_NAM}"

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        return db #yield db
    except:
        db.close()
