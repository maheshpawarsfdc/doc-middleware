from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
import requests
import os
import logging
from typing import List, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
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
    logger.info("🚀 FastAPI server starting up...")
    
    # Validate configuration
    if not os.environ.get('GROQ_API_KEY'):
        logger.error("❌ GROQ_API_KEY environment variable not set!")
        raise RuntimeError("GROQ_API_KEY is required")
    
    logger.info("✅ Configuration validated")
    yield
    
    # Shutdown
    logger.info("🛑 FastAPI server shutting down...")

app = FastAPI(
    title="Document Analysis Middleware", 
    version="2.0.0",
    lifespan=lifespan
)

# Configure requests session with retry strategy
def create_requests_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,  # Increased backoff
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

requests_session = create_requests_session()

# ----------------------------
# Models
# ----------------------------
class FileInput(BaseModel):
    filename: str
    filedata: str
    customPrompt: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class FollowupPayload(BaseModel):
    messages: List[Message]

# ----------------------------
# Configuration
# ----------------------------
MAX_TEXT_LENGTH = 80000  # Reduced to prevent token overflow
GROQ_TIMEOUT = 90  # Increased timeout
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB limit
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2

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
# /process Endpoint - IMPROVED
# ----------------------------
@app.post("/process")
async def process_file(input: FileInput):
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] 📩 Processing file: {input.filename}")
    
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

        # Build the prompt
        base_prompt = build_analysis_prompt(extracted_text)
        final_prompt = (
            input.customPrompt.strip() + "\n\n" + base_prompt
            if input.customPrompt else base_prompt
        )
        
        # Call AI service with proper error handling and retry
        response = await call_groq_with_retry(final_prompt, request_id)
        
        logger.info(f"[{request_id}] ✅ Analysis completed successfully")
        return {
            "insights": response, 
            "status": "success",
            "request_id": request_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ----------------------------
# Text Extraction Function - IMPROVED
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
# Prompt Building Function - SAME AS BEFORE
# ----------------------------
def build_analysis_prompt(extracted_text: str) -> str:
    """Build the analysis prompt with the extracted text"""
    return f"""
You are an expert legal and business document analysis assistant specialized in HR, Sales, and Legal document review.
**DOCUMENT CONTEXT:** Analyze this document as if you are reviewing it for a compliance team in a corporate environment. 
Focus on business-critical insights, regulatory compliance, and actionable recommendations, review everything and make a summary of everything that are the main/important things.

**ANALYSIS INSTRUCTIONS:**
1. First, identify the document type (contract, resume, NDA, policy, etc.)
2. Extract key information with high accuracy
3. Flag potential compliance issues or business risks
4. Provide specific, actionable recommendations
5. Use professional language suitable for business stakeholders
6. All important dates in the document, along with what they represent.
7. All important monetary amounts, along with what they represent and their currency if stated.


**OUTPUT FORMAT:** Return your analysis in this exact structure with proper line breaks:
---
**Document Type & Classification**
[Identify: Contract, Resume, NDA, Policy, Agreement, etc.]

**Document Summary**
[Provide a concise 3-5 sentence executive overview covering: purpose, key parties involved, main terms/conditions, and overall significance]

**Key Information Extracted**
**People & Roles:**
- [Name] - [Role/Title] - [Organization if mentioned]

**Organizations & Entities:**
- [Organization Name] - [Type: Company/Agency/etc.] - [Role in document]

**Important Dates:**
CRITICAL: Every date MUST follow this exact format: "[Label]: [Date]"
- [Date Label/Context]: [Date] - [Additional significance if any]
Examples: "Joining Date: 16 Aug 2025", "Contract End Date: 31 Dec 2025", "Notice Period Deadline: 15 Sep 2025"
DO NOT write dates like "16 Aug 2025" - ALWAYS include the label like "Joining Date: 16 Aug 2025"

**Monetary Values & Terms:**
CRITICAL: Every amount MUST follow this exact format: "[Label]: [Currency][Amount]"
- [Amount Label/Context]: [Currency][Amount] - [Additional details if relevant]
Examples: "Fixed CTC: ₹1,000,000 annually", "Variable Pay: ₹200,000 (20% of fixed)", "Joining Bonus: ₹100,000 (one-time)"
DO NOT write amounts like "1000000" - ALWAYS include the label like "Fixed CTC: ₹1,000,000"

**Critical Clauses & Terms:**
- [Brief description of key contractual terms, obligations, or conditions]

**Compliance & Risk Assessment**
**Potential Risks or Red Flags:**
- [HIGH/MEDIUM/LOW] [Specific risk with brief explanation]

**Missing or Unclear Elements:**
- [Items that should be present but are missing or ambiguous]

**Regulatory Considerations:**
- [Any compliance requirements, legal standards, or regulatory issues identified]

**Actionable Recommendations**
**Immediate Actions Required:**
- [Priority 1 items that need immediate attention]

**Follow-up Actions:**
- [Items to address within specific timeframes]

**Stakeholder Notifications:**
- [Who should be informed about this document and why]

**Document Management:**
- [Filing, renewal dates, or administrative actions needed]
---

**CRITICAL EXTRACTION GUIDELINES:**

**For Dates:** ALWAYS include the context/label before the date. NEVER list just raw dates.
- Format: "[Purpose/Label]: [Date] - [Additional context if relevant]"
- Examples: "Employment Start Date: 1 Jan 2025", "Probation End: 30 Jun 2025", "Annual Review Due: 31 Dec 2025"
- WRONG: "16 Aug 2025" 
- CORRECT: "Joining Date: 16 Aug 2025"

**For Monetary Values:** ALWAYS include what the amount represents. NEVER list just raw numbers.
- Format: "[Purpose/Label]: [Currency][Amount] - [Additional context if relevant]"
- Examples: "Base Salary: $75,000 annually", "Performance Bonus: $10,000 (quarterly)", "Severance Pay: $25,000 (3 months)"
- WRONG: "1000000"
- CORRECT: "Fixed CTC: ₹1,000,000 annually"

**For Currency:** Use appropriate currency symbols (₹, $, €, £) based on document context.

**MANDATORY RULES:**
1. NEVER extract dates or amounts without their contextual labels
2. If context is unclear, use descriptive labels like "Unspecified Date: [date]" or "Miscellaneous Amount: [amount]"
3. Every date MUST have a purpose/label explaining what it represents
4. Every monetary value MUST have a label explaining what it represents
5. If you cannot determine the context, explicitly state "Context unclear" but still provide a descriptive label

**QUALITY GUIDELINES:**
- Be specific and avoid generic statements
- Include confidence levels when uncertain (e.g., "appears to be" for unclear information)
- Focus on business impact and legal significance
- Prioritize risks by severity (HIGH/MEDIUM/LOW)
- Ensure all recommendations are actionable with clear next steps

**DOCUMENT CONTENT TO ANALYZE:**
{extracted_text}

**REMINDER: You MUST provide context labels for ALL dates and monetary values. Raw numbers and dates without context are NOT acceptable.**

**End of analysis request.**
"""

# ----------------------------
# /followup Endpoint - IMPROVED
# ----------------------------
@app.post("/followup")
async def followup_chat(payload: FollowupPayload):
    request_id = str(int(time.time() * 1000))
    logger.info(f"[{request_id}] 💬 Processing follow-up chat")
    
    try:
        if not payload.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Validate messages
        valid_messages = []
        for msg in payload.messages:
            if msg.role and msg.content:
                valid_messages.append(msg.dict())
            else:
                logger.warning(f"[{request_id}] Skipping invalid message: {msg}")
        
        if not valid_messages:
            raise HTTPException(status_code=400, detail="No valid messages found")
        
        response = await call_groq_messages_with_retry(valid_messages, request_id)
        logger.info(f"[{request_id}] ✅ Follow-up completed successfully")
        
        return {
            "reply": response, 
            "status": "success",
            "request_id": request_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Follow-up error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")

# ----------------------------
# Utility: Single prompt with retry - IMPROVED
# ----------------------------
async def call_groq_with_retry(prompt: str, request_id: str, max_retries: int = MAX_RETRIES) -> str:
    """Call Groq API with retry logic and proper error handling"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a professional document analysis assistant specializing in legal, HR, and business document review."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[{request_id}] 🤖 Calling Groq API (attempt {attempt + 1}/{max_retries})")
            
            response = requests_session.post(
                groq_api_url, 
                headers=headers, 
                json=payload,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"[{request_id}] ✅ Groq API successful, response length: {len(content)} chars")
                return content
            
            elif response.status_code == 429:  # Rate limit
                retry_after = response.headers.get('retry-after', INITIAL_RETRY_DELAY * (2 ** attempt))
                logger.warning(f"[{request_id}] ⏰ Rate limited, waiting {retry_after}s")
                await asyncio.sleep(float(retry_after))
                continue
                
            elif response.status_code in [500, 502, 503, 504]:  # Server errors
                logger.warning(f"[{request_id}] 🔄 Server error {response.status_code}, retrying...")
                await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
                continue
                
            else:
                error_msg = f"Groq API error {response.status_code}: {response.text}"
                logger.error(f"[{request_id}] ❌ {error_msg}")
                last_error = error_msg
                break
                    
        except requests.exceptions.Timeout:
            logger.error(f"[{request_id}] ⏰ Groq API timeout on attempt {attempt + 1}")
            last_error = "AI service timeout"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[{request_id}] 🌐 Network error on attempt {attempt + 1}: {str(e)}")
            last_error = f"Network error: {str(e)}"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Unexpected error on attempt {attempt + 1}: {str(e)}")
            last_error = f"AI service error: {str(e)}"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
    
    # All retries failed
    raise HTTPException(
        status_code=503, 
        detail=f"AI service unavailable after {max_retries} attempts. Last error: {last_error}"
    )

# ----------------------------
# Utility: Chat with message history and retry - IMPROVED
# ----------------------------
async def call_groq_messages_with_retry(messages: List[dict], request_id: str, max_retries: int = MAX_RETRIES) -> str:
    """Call Groq API with message history and retry logic"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[{request_id}] 💬 Calling Groq chat API (attempt {attempt + 1}/{max_retries})")
            
            response = requests_session.post(
                groq_api_url, 
                headers=headers, 
                json=payload,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"[{request_id}] ✅ Chat API successful")
                return content
                
            elif response.status_code == 429:  # Rate limit
                retry_after = response.headers.get('retry-after', INITIAL_RETRY_DELAY * (2 ** attempt))
                logger.warning(f"[{request_id}] ⏰ Chat rate limited, waiting {retry_after}s")
                await asyncio.sleep(float(retry_after))
                continue
                
            elif response.status_code in [500, 502, 503, 504]:  # Server errors
                logger.warning(f"[{request_id}] 🔄 Chat server error {response.status_code}, retrying...")
                await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
                continue
                
            else:
                error_msg = f"Chat API error {response.status_code}: {response.text}"
                logger.error(f"[{request_id}] ❌ {error_msg}")
                last_error = error_msg
                break
                
        except requests.exceptions.Timeout:
            logger.error(f"[{request_id}] ⏰ Chat API timeout on attempt {attempt + 1}")
            last_error = "Chat service timeout"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[{request_id}] 🌐 Chat network error on attempt {attempt + 1}: {str(e)}")
            last_error = f"Network error: {str(e)}"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Chat unexpected error on attempt {attempt + 1}: {str(e)}")
            last_error = f"Chat service error: {str(e)}"
            await asyncio.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
    
    # All retries failed
    raise HTTPException(
        status_code=503, 
        detail=f"Chat service unavailable after {max_retries} attempts. Last error: {last_error}"
    )

# ----------------------------
# Health Check Endpoint - IMPROVED
# ----------------------------
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "groq_api_configured": bool(os.environ.get('GROQ_API_KEY')),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_text_length": MAX_TEXT_LENGTH,
        "timeout_seconds": GROQ_TIMEOUT
    }
    
    # Test Groq API connectivity (optional)
    try:
        groq_api_key = os.environ.get('GROQ_API_KEY')
        if groq_api_key:
            test_response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {groq_api_key}"},
                timeout=5
            )
            health_status["groq_api_accessible"] = test_response.status_code == 200
        else:
            health_status["groq_api_accessible"] = False
    except Exception as e:
        logger.warning(f"Health check API test failed: {str(e)}")
        health_status["groq_api_accessible"] = False
    
    return health_status

# ----------------------------
# Info Endpoint
# ----------------------------
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Document Analysis Middleware",
        "version": "2.0.0",
        "description": "FastAPI middleware for document analysis using AI",
        "endpoints": {
            "/process": "POST - Analyze document files",
            "/followup": "POST - Follow-up chat questions",
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
