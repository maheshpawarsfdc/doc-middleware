from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
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
    logger.info("🚀 Document Conversion Middleware starting up...")
    logger.info("✅ Middleware ready for document conversion")
    yield
    
    # Shutdown
    logger.info("🛑 Document Conversion Middleware shutting down...")

app = FastAPI(
    title="Document Conversion Middleware", 
    version="3.0.0",
    description="FastAPI middleware for document-to-text conversion only",
    lifespan=lifespan
)

# ----------------------------
# Models
# ----------------------------
class ConversionRequest(BaseModel):
    filename: str
    filedata: str  # base64 encoded

class ConversionResponse(BaseModel):
    extracted_text: str
    filename: str
    file_size_bytes: int
    pages_processed: Optional[int] = None
    paragraphs_processed: Optional[int] = None
    conversion_time_ms: int
    status: str
    request_id: str

# ----------------------------
# Configuration
# ----------------------------
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB limit
MAX_TEXT_LENGTH = 100000  # 100K characters max

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
# /convert Endpoint - MAIN CONVERSION ENDPOINT
# ----------------------------
@app.post("/convert", response_model=ConversionResponse)
async def convert_document(request: ConversionRequest):
    request_id = str(int(time.time() * 1000))
    start_time = time.time()
    logger.info(f"[{request_id}] 📄 Converting document: {request.filename}")
    
    try:
        # Input validation
        if not request.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        if not request.filedata:
            raise HTTPException(status_code=400, detail="File data is required")
        
        # Validate file type
        supported_extensions = ['.pdf', '.docx']
        file_extension = None
        for ext in supported_extensions:
            if request.filename.lower().endswith(ext):
                file_extension = ext
                break
        
        if not file_extension:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Decode and validate file data
        try:
            binary_data = base64.b64decode(request.filedata)
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Base64 decode error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 file data")
        
        # Check file size
        if len(binary_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
            )
        
        logger.info(f"[{request_id}] 📊 File size: {len(binary_data)} bytes")
        
        # Extract text based on file type
        extraction_result = await extract_text_from_file(
            request.filename, 
            binary_data, 
            file_extension,
            request_id
        )
        
        extracted_text = extraction_result['text']
        pages_processed = extraction_result.get('pages')
        paragraphs_processed = extraction_result.get('paragraphs')
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=422, 
                detail="No text could be extracted from the document"
            )

        # Truncate if too long
        original_length = len(extracted_text)
        if len(extracted_text) > MAX_TEXT_LENGTH:
            logger.warning(f"[{request_id}] ⚠️ Text truncated from {original_length} to {MAX_TEXT_LENGTH} characters")
            extracted_text = extracted_text[:MAX_TEXT_LENGTH] + "\n\n[Document truncated due to length...]"
        
        conversion_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"[{request_id}] ✅ Conversion completed in {conversion_time}ms - {len(extracted_text)} characters extracted")
        
        return ConversionResponse(
            extracted_text=extracted_text,
            filename=request.filename,
            file_size_bytes=len(binary_data),
            pages_processed=pages_processed,
            paragraphs_processed=paragraphs_processed,
            conversion_time_ms=conversion_time,
            status="success",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Conversion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document conversion failed: {str(e)}")

# ----------------------------
# Text Extraction Function - UPDATED
# ----------------------------
async def extract_text_from_file(filename: str, binary_data: bytes, file_extension: str, request_id: str) -> dict:
    """Extract text from PDF or DOCX files - returns dict with text and metadata"""
    extracted_text = ""
    pages_processed = None
    paragraphs_processed = None
    
    try:
        if file_extension == '.pdf':
            logger.info(f"[{request_id}] 📄 Processing PDF file")
            
            try:
                with pdfplumber.open(BytesIO(binary_data)) as pdf:
                    total_pages = len(pdf.pages)
                    pages_processed = total_pages
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
                        
        elif file_extension == '.docx':
            logger.info(f"[{request_id}] 📄 Processing DOCX file")
            
            try:
                doc = Document(BytesIO(binary_data))
                total_paragraphs = len(doc.paragraphs)
                paragraphs_processed = total_paragraphs
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
        'pages': pages_processed,
        'paragraphs': paragraphs_processed
    }

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for conversion service"""
    return {
        "status": "healthy",
        "service": "Document Conversion Middleware",
        "version": "3.0.0",
        "timestamp": time.time(),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_text_length": MAX_TEXT_LENGTH,
        "supported_formats": ["PDF", "DOCX"]
    }

# ----------------------------
# Info Endpoint
# ----------------------------
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Document Conversion Middleware",
        "version": "3.0.0",
        "description": "FastAPI middleware for document-to-text conversion only",
        "endpoints": {
            "/convert": "POST - Convert document to text",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "supported_formats": ["PDF", "DOCX"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "purpose": "Document conversion only - AI processing handled by client"
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
