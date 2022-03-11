from fastapi import FastAPI, Depends, HTTPException, APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from .auth_handler import AuthHandler
from src.app.main import *
from io import BytesIO

auth_handler = AuthHandler()
router = APIRouter()

@router.get("/process-pdf-unprotected")
async def upload_file_unprotected(file: UploadFile = File(...)):
    content = await file.read()
    zpath = process_pdf_stream(content, file.filename )
    with open(zpath,"rb") as f: bstring = f.read()
    io = BytesIO(bstring)
    return StreamingResponse(iter([io.getvalue()]),
                             media_type = "application/x-zip-compressed",
                             headers = { f"Content-Disposition":f"attachment;filename={file.filename}" } )

@router.post("/process-pdf")
async def upload_file(file: UploadFile = File(...), username=Depends(auth_handler.auth_wrapper)):
    content = await file.read()
    fpath = process_pdf_stream(content, file.filename, username )
    with open(zpath,"rb") as f: bstring = f.read()
    io = BytesIO(bstring)
    return StreamingResponse(iter([io.getvalue()]),
                             media_type="application/x-zip-compressed",
                             headers = { f"Content-Disposition":f"attachment;filename={file.filename}"})
