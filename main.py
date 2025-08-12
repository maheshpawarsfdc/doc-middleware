from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
import os
import logging
from typing import Optional
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 FastAPI Document Processing Middleware starting up...")
    logger.info("✅ Configuration validated")
    yield
    
    # Shutdown
    logger.info("🛑 FastAPI server shutting down...")

app = FastAPI(
    title="Document Processing Middleware", 
    version="1.0.0",
    lifespan=lifespan
)

# ----------------------------
# Models
# ----------------------------
class FileInput(BaseModel):
    filename: str
    filedata: str

class ProcessedDocument(BaseModel):
    filename: str
    extracted_text: str
    status: str
    request_id: str
    timestamp: float

# ----------------------------
# Configuration
# ----------------------------
MAX_TEXT_LENGTH = 80000  # Maximum text length to prevent token overflow
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB limit

# ----------------------------
# Error Handlers
# ----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error occurred",
        "status_code": 500,
        "timestamp": time.time()
    }

# ----------------------------
# /process Endpoint - Document Processing Only
# ----------------------------
@app.post("/process")
async def process_document(input: FileInput):
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] 📩 Processing document: {input.filename}")
    
    try:
        # Input validation
        if not input.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        if not input.filedata:
            raise HTTPException(status_code=400, detail="File data is required")
        
        # Decode and validate file data
        try:
            binary = base64.b64decode(input.filedata)
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Base64 decode error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 file data")
        
        # Check file size
        if len(binary) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
            )
        
        logger.info(f"[{request_id}] 📊 File size: {len(binary)} bytes")
        
        # Extract text based on file type
        extracted_text = await extract_text_from_file(input.filename, binary, request_id)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")

        # Truncate if too long
        if len(extracted_text) > MAX_TEXT_LENGTH:
            logger.warning(f"[{request_id}] ⚠️ Text truncated from {len(extracted_text)} to {MAX_TEXT_LENGTH} characters")
            extracted_text = extracted_text[:MAX_TEXT_LENGTH] + "\n\n[Document truncated due to length...]"
        
        logger.info(f"[{request_id}] 📄 Extracted {len(extracted_text)} characters of text")

        logger.info(f"[{request_id}] ✅ Document processing completed successfully")
        
        return ProcessedDocument(
            filename=input.filename,
            extracted_text=extracted_text,
            status="success",
            request_id=request_id,
            timestamp=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ----------------------------
# Text Extraction Function
# ----------------------------
async def extract_text_from_file(filename: str, binary: bytes, request_id: str) -> str:
    """Extract text from PDF or DOCX files with improved error handling"""
    extracted_text = ""
    
    try:
        if filename.lower().endswith(".pdf"):
            logger.info(f"[{request_id}] 📄 Processing PDF file")
            
            try:
                with pdfplumber.open(BytesIO(binary)) as pdf:
                    total_pages = len(pdf.pages)
                    logger.info(f"[{request_id}] 📊 PDF has {total_pages} pages")
                    
                    if total_pages == 0:
                        raise ValueError("PDF file appears to be empty or corrupted")
                    
                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"
                            logger.debug(f"[{request_id}] ✅ Processed page {i+1}/{total_pages}")
                        except Exception as e:
                            logger.warning(f"[{request_id}] ⚠️ Error extracting page {i+1}: {str(e)}")
                            continue
            except Exception as e:
                logger.error(f"[{request_id}] ❌ PDF processing failed: {str(e)}")
                raise HTTPException(status_code=422, detail=f"PDF processing failed: {str(e)}")
                        
        elif filename.lower().endswith(".docx"):
            logger.info(f"[{request_id}] 📄 Processing DOCX file")
            
            try:
                doc = Document(BytesIO(binary))
                total_paragraphs = len(doc.paragraphs)
                logger.info(f"[{request_id}] 📊 DOCX has {total_paragraphs} paragraphs")
                
                if total_paragraphs == 0:
                    raise ValueError("DOCX file appears to be empty or corrupted")
                
                for i, para in enumerate(doc.paragraphs):
                    try:
                        if para.text.strip():
                            extracted_text += para.text + "\n"
                    except Exception as e:
                        logger.warning(f"[{request_id}] ⚠️ Error extracting paragraph {i+1}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"[{request_id}] ❌ DOCX processing failed: {str(e)}")
                raise HTTPException(status_code=422, detail=f"DOCX processing failed: {str(e)}")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Only PDF and DOCX are supported."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Text extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    
    return extracted_text.strip()

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "service": "Document Processing Middleware",
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_text_length": MAX_TEXT_LENGTH
    }

# ----------------------------
# Info Endpoint
# ----------------------------
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Document Processing Middleware",
        "version": "1.0.0",
        "description": "FastAPI middleware for document text extraction",
        "endpoints": {
            "/process": "POST - Extract text from document files",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "supported_formats": ["PDF", "DOCX"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
