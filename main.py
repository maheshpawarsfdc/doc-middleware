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
    version="2.1.0",
    lifespan=lifespan
)

# Configure requests session with retry strategy
def create_requests_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
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
MAX_TEXT_LENGTH = 80000
GROQ_TIMEOUT = 120  # Increased timeout for comprehensive analysis
MAX_FILE_SIZE = 15 * 1024 * 1024
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
# /process Endpoint
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

        # Build the ENHANCED prompt with summary focus
        final_prompt = build_comprehensive_analysis_prompt(extracted_text, input.customPrompt)
        
        # Call AI service with proper error handling and retry
        response = await call_groq_with_retry(final_prompt, request_id)
        
        # ENHANCED: Validate summary quality before returning
        summary_quality = validate_summary_quality(response, request_id)
        
        logger.info(f"[{request_id}] ✅ Analysis completed successfully - Summary quality: {summary_quality}")
        return {
            "insights": response, 
            "status": "success",
            "request_id": request_id,
            "timestamp": time.time(),
            "summary_quality": summary_quality
        }
        
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
# COMPREHENSIVE Prompt Building Function
# ----------------------------
def build_comprehensive_analysis_prompt(extracted_text: str, custom_prompt: Optional[str] = None) -> str:
    """Build comprehensive analysis prompt with extreme focus on detailed summaries"""
    
    base_prompt = f"""
🚨 EXECUTIVE SUMMARY PRIORITY 🚨
You are a senior business analyst creating executive-level document briefings. The DOCUMENT SUMMARY is your PRIMARY DELIVERABLE and must be exceptionally detailed and comprehensive.

**CRITICAL SUCCESS CRITERIA:**
✅ Summary must be 10-12 complete sentences (not bullet points)
✅ Summary must be so detailed that executives never need to read the original document
✅ Every important detail, date, amount, and term must be included in the summary
✅ Written in flowing, professional prose suitable for C-level executives

**MANDATORY FORMATTING RULES:**

1. **DATES:** Always provide context - NEVER list raw dates
   ❌ WRONG: "January 15, 2024"
   ✅ CORRECT: "Employment Start Date: January 15, 2024"

2. **MONETARY VALUES:** Always provide context - NEVER list raw amounts  
   ❌ WRONG: "$185,000"
   ✅ CORRECT: "Annual Base Salary: $185,000"

3. **NO SHORTCUTS:** Every number, date, and amount must have descriptive context

**REQUIRED OUTPUT FORMAT:**

**Document Type & Classification**
[Precisely identify the document type and its business purpose]

**Document Summary**
[CRITICAL: This is your most important section. Write exactly 10-12 comprehensive sentences in flowing prose that capture EVERY important detail from the document. Include all parties, all amounts, all dates, all terms, all risks, and all recommendations. Make this so complete that the reader never needs to see the original document.]

**Key Information Extracted**

**People & Roles:**
- [Full Name] - [Complete Title] - [Organization] - [Role in document]

**Organizations & Entities:**  
- [Organization Name] - [Type] - [Specific role/relationship in document]

**Important Dates:**
- [Descriptive Label]: [Date] - [Business significance/context]

**Monetary Values & Terms:**
- [Descriptive Label]: [Amount] - [Payment terms/frequency/conditions]

**Critical Clauses & Terms:**
- [Key contractual terms, obligations, restrictions, benefits]

**Compliance & Risk Assessment**

**Potential Risks or Red Flags:**
- [HIGH/MEDIUM/LOW] [Specific risk with detailed explanation]

**Missing or Unclear Elements:**
- [Important items that should be present but are missing or ambiguous]

**Regulatory Considerations:**
- [Compliance requirements, legal standards, regulatory issues]

**Actionable Recommendations**

**Immediate Actions Required:**
- [Priority actions with specific timelines]

**Follow-up Actions:**
- [Secondary actions with recommended timeframes]

**Stakeholder Notifications:**
- [Who needs to be informed and why]

**Document Management:**
- [Filing, renewal, administrative requirements]

**EXAMPLE OF PERFECT SUMMARY:**
"This employment agreement establishes John Smith as Senior Software Engineer at TechCorp Industries effective January 15, 2025, reporting directly to the VP of Engineering with responsibility for mobile development team leadership. The compensation package includes an annual base salary of $185,000, performance bonus potential up to $25,000, stock options valued at $50,000 vesting over four years, and a one-time $10,000 relocation allowance. The agreement specifies a 90-day probationary period ending April 15, 2025, during which termination requires only 24-hour notice from either party. Key benefits include immediate health insurance coverage, 20 vacation days annually, and access to company fitness facilities. The contract contains a 12-month non-compete restriction covering technology companies within 50 miles and confidentiality obligations extending two years post-employment. Annual performance reviews are scheduled for January 15th with salary adjustment consideration based on company performance metrics. The agreement automatically renews annually unless either party provides 30-day written notice of termination. Notable concerns include broad non-compete language that may significantly limit future employment opportunities and absence of specific intellectual property assignment clauses. Missing elements include detailed vacation accrual policies, sick leave provisions, and specific performance bonus calculation methods. This represents a standard corporate employment agreement requiring routine HR processing, employee handbook acknowledgment, and IT system access setup."

**ANALYSIS TARGET:**
{extracted_text}

**FINAL REMINDER:** The summary must be comprehensive enough that reading the original document becomes unnecessary for business decisions. Include ALL specific details, amounts, dates, and terms in flowing, executive-level prose.
"""

    if custom_prompt:
        return f"{custom_prompt.strip()}\n\n{base_prompt}"
    return base_prompt

# ----------------------------
# NEW: Summary Quality Validation
# ----------------------------
def validate_summary_quality(response: str, request_id: str) -> str:
    """Validate the quality and completeness of the generated summary"""
    try:
        # Extract the summary section
        summary_start = response.find("**Document Summary**")
        if summary_start == -1:
            logger.warning(f"[{request_id}] ⚠️ No Document Summary section found")
            return "missing"
        
        # Find the end of summary section
        summary_end = response.find("**Key Information Extracted**", summary_start)
        if summary_end == -1:
            summary_end = len(response)
        
        summary_content = response[summary_start:summary_end].replace("**Document Summary**", "").strip()
        
        # Count sentences
        sentences = [s.strip() for s in summary_content.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Check length
        word_count = len(summary_content.split())
        
        logger.info(f"[{request_id}] 📊 Summary validation - Sentences: {sentence_count}, Words: {word_count}")
        
        # Quality assessment
        if sentence_count >= 10 and word_count >= 200:
            return "excellent"
        elif sentence_count >= 8 and word_count >= 150:
            return "good"
        elif sentence_count >= 5 and word_count >= 100:
            return "acceptable"
        else:
            logger.warning(f"[{request_id}] ⚠️ Summary quality concerns - only {sentence_count} sentences, {word_count} words")
            return "needs_improvement"
            
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error validating summary: {str(e)}")
        return "validation_error"

# ----------------------------
# /followup Endpoint
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
# Utility: Single prompt with retry
# ----------------------------
async def call_groq_with_retry(prompt: str, request_id: str, max_retries: int = MAX_RETRIES) -> str:
    """Call Groq API with retry logic and enhanced summary focus"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system", 
                "content": "You are a senior business analyst specializing in executive document summaries. Your PRIMARY OBJECTIVE is creating comprehensive 10-12 sentence summaries that eliminate the need to read original documents. CRITICAL REQUIREMENTS: (1) Every summary must be exactly 10-12 sentences in flowing prose (2) Include ALL specific details: dates, amounts, parties, terms (3) Use descriptive labels for ALL dates and monetary values - NEVER raw numbers (4) Write for C-level executives making business decisions (5) Summary must be so complete that original document becomes unnecessary. Focus intensely on summary quality - this is your most important deliverable."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.05,  # Very low temperature for consistent, detailed formatting
        "max_tokens": 5000,   # Increased for comprehensive summaries
        "top_p": 0.9         # Added for more focused responses
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
# Utility: Chat with message history and retry
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
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.1.0",
        "groq_api_configured": bool(os.environ.get('GROQ_API_KEY')),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "max_text_length": MAX_TEXT_LENGTH,
        "timeout_seconds": GROQ_TIMEOUT,
        "summary_focus": "enhanced"
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
        "version": "2.1.0",
        "description": "FastAPI middleware for comprehensive document analysis with enhanced executive summary focus",
        "features": [
            "Enhanced 10-12 sentence executive summaries",
            "Comprehensive document analysis", 
            "Risk assessment and compliance checking",
            "Summary quality validation",
            "Retry logic and error handling"
        ],
        "endpoints": {
            "/process": "POST - Analyze document files with comprehensive summaries",
            "/followup": "POST - Follow-up chat questions",
            "/health": "GET - Health check with summary quality metrics",
            "/docs": "GET - API documentation"
        },
        "supported_formats": ["PDF", "DOCX"],
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "summary_requirements": "10-12 comprehensive sentences for executive decision-making"
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
