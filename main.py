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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Analysis Middleware", version="1.0.0")

# Configure requests session with retry strategy
def create_requests_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
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
MAX_TEXT_LENGTH = 100000  # Limit text to prevent token overflow
GROQ_TIMEOUT = 60  # 60 second timeout
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# ----------------------------
# /process Endpoint
# ----------------------------
@app.post("/process")
def process_file(input: FileInput):
    try:
        filename = input.filename
        logger.info(f"📩 Processing file: {filename}")
        
        # Decode and validate file data
        try:
            binary = base64.b64decode(input.filedata)
        except Exception as e:
            logger.error(f"❌ Base64 decode error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 file data")
        
        # Check file size
        if len(binary) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB")
        
        logger.info(f"📊 File size: {len(binary)} bytes")
        
        # Extract text based on file type
        extracted_text = extract_text_from_file(filename, binary)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")

        # Truncate if too long
        if len(extracted_text) > MAX_TEXT_LENGTH:
            logger.warning(f"⚠️ Text truncated from {len(extracted_text)} to {MAX_TEXT_LENGTH} characters")
            extracted_text = extracted_text[:MAX_TEXT_LENGTH] + "\n\n[Document truncated due to length...]"
        
        logger.info(f"📄 Extracted {len(extracted_text)} characters of text")

        # Build the prompt
        base_prompt = build_analysis_prompt(extracted_text)
        final_prompt = (
            input.customPrompt.strip() + "\n\n" + base_prompt
            if input.customPrompt else base_prompt
        )
        
        # Call AI service with proper error handling
        response = call_groq_with_retry(final_prompt)
        
        logger.info("✅ Analysis completed successfully")
        return {"insights": response, "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# ----------------------------
# Text Extraction Function
# ----------------------------
def extract_text_from_file(filename: str, binary: bytes) -> str:
    """Extract text from PDF or DOCX files with error handling"""
    extracted_text = ""
    
    try:
        if filename.lower().endswith(".pdf"):
            logger.info("📄 Processing PDF file")
            with pdfplumber.open(BytesIO(binary)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"📊 PDF has {total_pages} pages")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                        logger.debug(f"✅ Processed page {i+1}/{total_pages}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error extracting page {i+1}: {str(e)}")
                        continue
                        
        elif filename.lower().endswith(".docx"):
            logger.info("📄 Processing DOCX file")
            doc = Document(BytesIO(binary))
            total_paragraphs = len(doc.paragraphs)
            logger.info(f"📊 DOCX has {total_paragraphs} paragraphs")
            
            for i, para in enumerate(doc.paragraphs):
                try:
                    if para.text.strip():
                        extracted_text += para.text + "\n"
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting paragraph {i+1}: {str(e)}")
                    continue
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX are supported.")
            
    except Exception as e:
        logger.error(f"❌ Text extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    
    return extracted_text.strip()

# ----------------------------
# Prompt Building Function
# ----------------------------
def build_analysis_prompt(extracted_text: str) -> str:
    """Build the analysis prompt with the extracted text"""
    return f"""
You are an expert legal and business document analysis assistant specialized in HR, Sales, and Legal document review.
**DOCUMENT CONTEXT:** Analyze this document as if you are reviewing it for a compliance team in a corporate environment. Focus on business-critical insights, regulatory compliance, and actionable recommendations.
**ANALYSIS INSTRUCTIONS:**
1. First, identify the document type (contract, resume, NDA, policy, etc.)
2. Extract key information with high accuracy
3. Flag potential compliance issues or business risks
4. Provide specific, actionable recommendations
5. Use professional language suitable for business stakeholders

**OUTPUT FORMAT:** Return your analysis in this exact structure with proper line breaks:
---
**Document Type & Classification**
[Identify: Contract, Resume, NDA, Policy, Agreement, etc.]
**Document Summary**
[Provide a concise 5-8 sentence executive overview covering: purpose, key parties involved, main terms/conditions, and overall significance]
**Key Information Extracted**
**People & Roles:**
- [Name] - [Role/Title] - [Organization if mentioned]
**Organizations & Entities:**
- [Organization Name] - [Type: Company/Agency/etc.] - [Role in document]
**Important Dates:**
- - Format: "[Purpose/Label]: [Date] - [Additional context if relevant]"
- Examples: "Employment Start Date: 1 Jan 2025", "Probation End: 30 Jun 2025"
**Monetary Values & Terms:**
- - Format: "[Purpose/Label]: [Currency][Amount] - [Additional context if relevant]" 
- Examples: "Base Salary: $75,000 annually", "Performance Bonus: $10,000 (quarterly)"
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
**For Dates:** Always include the context/label before the date. Never list just raw dates.
- Format: "[Purpose/Label]: [Date] - [Additional context if relevant]"
- Examples: "Employment Start Date: 1 Jan 2025", "Probation End: 30 Jun 2025", "Annual Review Due: 31 Dec 2025"

**For Monetary Values:** Always include what the amount represents. Never list just raw numbers.
- Format: "[Purpose/Label]: [Currency][Amount] - [Additional context if relevant]"
- Examples: "Base Salary: $75,000 annually", "Performance Bonus: $10,000 (quarterly)", "Severance Pay: $25,000 (3 months)"

**For Currency:** Use appropriate currency symbols (₹, $, €, £) based on document context.

**QUALITY GUIDELINES:**
- Be specific and avoid generic statements
- Include confidence levels when uncertain (e.g., "appears to be" for unclear information)
- Focus on business impact and legal significance
- Prioritize risks by severity (HIGH/MEDIUM/LOW)
- Ensure all recommendations are actionable with clear next steps
- NEVER extract dates or amounts without their contextual labels
- If context is unclear, use descriptive labels like "Unspecified Date" or "Miscellaneous Amount"
**DOCUMENT CONTENT TO ANALYZE:**
{extracted_text}
**End of analysis request.**
"""

# ----------------------------
# /followup Endpoint
# ----------------------------
@app.post("/followup")
def followup_chat(payload: FollowupPayload):
    try:
        logger.info("💬 Processing follow-up chat")
        messages = [msg.dict() for msg in payload.messages]
        response = call_groq_messages_with_retry(messages)
        logger.info("✅ Follow-up completed successfully")
        return {"reply": response, "status": "success"}
    except Exception as e:
        logger.error(f"❌ Follow-up error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")

# ----------------------------
# Utility: Single prompt with retry
# ----------------------------
def call_groq_with_retry(prompt: str, max_retries: int = 3) -> str:
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
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 Calling Groq API (attempt {attempt + 1}/{max_retries})")
            
            response = requests_session.post(
                groq_api_url, 
                headers=headers, 
                json=payload,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"✅ Groq API successful, response length: {len(content)} chars")
                return content
            else:
                error_msg = f"Groq API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                if response.status_code == 429:  # Rate limit
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                elif attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=error_msg)
                    
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Groq API timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="AI service timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Network error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="All retry attempts failed")

# ----------------------------
# Utility: Chat with message history and retry
# ----------------------------
def call_groq_messages_with_retry(messages: List[dict], max_retries: int = 3) -> str:
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
    
    for attempt in range(max_retries):
        try:
            response = requests_session.post(
                groq_api_url, 
                headers=headers, 
                json=payload,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                if response.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"API error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise Exception("API timeout")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
    
    raise Exception("All retry attempts failed")

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "groq_api_configured": bool(os.environ.get('GROQ_API_KEY'))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
