from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
import requests
import os
from typing import List

app = FastAPI()

# ----------------------------
# Models
# ----------------------------
class FileInput(BaseModel):
    filename: str
    filedata: str
    customPrompt: str | None = None

class Message(BaseModel):
    role: str
    content: str

class FollowupPayload(BaseModel):
    messages: List[Message]

# ----------------------------
# /process Endpoint
# ----------------------------
@app.post("/process")
def process_file(input: FileInput):
    filename = input.filename
    binary = base64.b64decode(input.filedata)
    extracted_text = ""

    print(f"📩 Received file: {filename}")

    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(BytesIO(binary)) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""
    elif filename.lower().endswith(".docx"):
        doc = Document(BytesIO(binary))
        for para in doc.paragraphs:
            extracted_text += para.text + "\n"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

    print(f"📄 Extracted {len(extracted_text)} characters of text")

    base_prompt = f"""
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
[Provide a concise 3-5 sentence executive overview covering: purpose, key parties involved, main terms/conditions, and overall significance]

**Key Information Extracted**

**People & Roles:**
- [Name] - [Role/Title] - [Organization if mentioned]

**Organizations & Entities:**
- [Organization Name] - [Type: Company/Agency/etc.] - [Role in document]

**Important Dates:**
- [Date] - [Significance: Effective date, Expiration, Deadline, etc.]

**Monetary Values & Terms:**
- [Amount] - [Context: Salary, Fee, Penalty, Budget, etc.]

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

**QUALITY GUIDELINES:**
- Be specific and avoid generic statements
- Include confidence levels when uncertain (e.g., "appears to be" for unclear information)
- Focus on business impact and legal significance
- Prioritize risks by severity (HIGH/MEDIUM/LOW)
- Ensure all recommendations are actionable with clear next steps

**DOCUMENT CONTENT TO ANALYZE:**

{extracted_text}

**End of analysis request.**
"""

    final_prompt = (
        input.customPrompt.strip() + "\n\n" + base_prompt
        if input.customPrompt else base_prompt
    )

    response = call_groq(final_prompt)
    return {"insights": response}

# ----------------------------
# /followup Endpoint
# ----------------------------
@app.post("/followup")
def followup_chat(payload: FollowupPayload):
    try:
        messages = [msg.dict() for msg in payload.messages]
        response = call_groq_messages(messages)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Utility: Single prompt
# ----------------------------
def call_groq(prompt: str) -> str:
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a document analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    res = requests.post(groq_api_url, headers=headers, json=payload)
    if res.status_code != 200:
        raise Exception(res.text)
    print("✅ Groq returned a response.")
    return res.json()["choices"][0]["message"]["content"]

# ----------------------------
# Utility: Chat with message history
# ----------------------------
def call_groq_messages(messages: List[dict]) -> str:
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.2
    }
    res = requests.post(groq_api_url, headers=headers, json=payload)
    if res.status_code != 200:
        raise Exception(res.text)
    print("💬 Follow-up handled successfully.")
    return res.json()["choices"][0]["message"]["content"]
