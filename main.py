from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
import os
import logging
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
    logger.info("🚀 FastAPI Text Extraction Service starting up...")
    yield
    
    # Shutdown
    logger.info("🛑 FastAPI Text Extraction Service shutting down...")

app = FastAPI(
    title="Document Text Extraction Service", 
    version="3.0.0",
    description="Middleware service for extracting text from PDF and DOCX files",
    lifespan=lifespan
)

# ----------------------------
# Models
# ----------------------------
class FileInput(BaseModel):
    filename: str
    filedata: str

class TextExtractionResponse(BaseModel):
    extractedText: str
    filename: str
    fileSize: int
    textLength: int
    pages: int = 0
    paragraphs: int = 0
    status: str
    request_id: str
    timestamp: float

# ----------------------------
# Configuration
# ----------------------------
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
# /extract-text Endpoint - NEW
# ----------------------------
@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(input: FileInput):
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] 📄 Extracting text from file: {input.filename}")
    
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
        
        # Extract text and metadata
        extraction_result = await extract_text_from_file(input.filename, binary, request_id)
        
        if not extraction_result['text'].strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        logger.info(f"[{request_id}] ✅ Text extraction completed successfully")
        
        return TextExtractionResponse(
            extractedText=extraction_result['text'],
            filename=input.filename,
            fileSize=len(binary),
            textLength=len(extraction_result['text']),
            pages=extraction_result.get('pages', 0),
            paragraphs=extraction_result.get('paragraphs', 0),
            status="success",
            request_id=request_id,
            timestamp=time.time()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

# ----------------------------
# Text Extraction Function - IMPROVED
# ----------------------------
async def extract_text_from_file(filename: str, binary: bytes, request_id: str) -> dict:
    """Extract text from PDF or DOCX files with improved error handling"""
    extracted_text = ""
    metadata = {
        'pages': 0,
        'paragraphs': 0
    }
    
    try:
        if filename.lower().endswith(".pdf"):
            logger.info(f"[{request_id}] 📄 Processing PDF file")
            
            try:
                with pdfplumber.open(BytesIO(binary)) as pdf:
                    total_pages = len(pdf.pages)
                    metadata['pages'] = total_pages
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
                metadata['paragraphs'] = total_paragraphs
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
    
    return {
        'text': extracted_text.strip(),
        'pages': metadata['pages'],
        'paragraphs': metadata['paragraphs']
    }

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0",
        "service": "Text Extraction Only",
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "supported_formats": ["PDF", "DOCX"]
    }

# ----------------------------
# Info Endpoint
# ----------------------------
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Document Text Extraction Service",
        "version": "3.0.0",
        "description": "FastAPI service for extracting text from documents (PDF, DOCX)",
        "endpoints": {
            "/extract-text": "POST - Extract text from document files",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "supported_formats": ["PDF", "DOCX"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "note": "This service only extracts text. AI processing should be handled separately."
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
