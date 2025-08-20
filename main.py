from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import fitz  # PyMuPDF for PDF
import docx2txt
from io import BytesIO
import tempfile
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="DocToText Middleware", version="1.0.0")

# Request model
class FileRequest(BaseModel):
    fileName: str
    fileContent: str  # base64 encoded file

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using docx2txt"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        text = docx2txt.process(tmp_file_path) or ""
        os.remove(tmp_file_path)  # cleanup
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract text from DOCX")

@app.post("/extract-text")
async def extract_text(request: FileRequest):
    """Convert PDF/DOCX (base64) into plain text"""
    try:
        logger.info(f"Received file: {request.fileName}")

        # Decode base64
        file_bytes = base64.b64decode(request.fileContent)

        # Determine file type
        if request.fileName.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_bytes)
        elif request.fileName.lower().endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX are supported.")

        if not extracted_text:
            raise HTTPException(status_code=422, detail="No readable text found in the document")

        return {"fileName": request.fileName, "extractedText": extracted_text}

    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
