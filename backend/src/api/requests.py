from fastapi import FastAPI, Depends, HTTPException, APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from .auth_handler import AuthHandler
from src.app.main import *
from io import BytesIO

auth_handler = AuthHandler()
router = APIRouter()

@router.get("/process-pdf-unprotected") # testing
async def upload_file_unprotected(file: UploadFile = File(...)):
    content = await file.read()
    fname = file.filename.split(".")[0]
    zpath = process_pdf_stream(content, fname )
    with open(zpath,"rb") as f: bstring = f.read()
    io = BytesIO(bstring)
    os.remove(zpath)
    return StreamingResponse(iter([io.getvalue()]),
                             media_type = "application/x-zip-compressed",
                             headers = { f"Content-Disposition":f"attachment;filename={fname}" } )

@router.get("/process-pdf")
async def upload_file(file: UploadFile = File(...), username=Depends(auth_handler.auth_wrapper)):
    content = await file.read()
    fname = file.filename.split(".")[0]
    zpath = process_pdf_stream(content, fname, username )
    with open(zpath,"rb") as f: bstring = f.read()
    io = BytesIO(bstring)
    os.remove(zpath)
    return StreamingResponse(iter([io.getvalue()]),
                             media_type="application/x-zip-compressed",
                             headers = { f"Content-Disposition":f"attachment;filename={fname}"})
