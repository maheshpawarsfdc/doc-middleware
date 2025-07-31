from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pdfplumber
from docx import Document
from io import BytesIO
import requests
import os

app = FastAPI()

class FileInput(BaseModel):
    filename: str
    filedata: str
    customPrompt: str | None = None
    
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

    prompt = f"""
You are a legal and business document analysis assistant.

Analyze the following document text and return insights in this **exact format** with each section starting on a **new line**, and make sure to include **line breaks between sections and list items**:

---

**Document Summary**  
[A concise 3–5 sentence overview of what the document is about.]

**Named Entities**  
People:  
- [list of people]  
Organizations:  
- [list of orgs]  
Dates:  
- [list of dates]  
Monetary Values:  
- [list of monetary values]

**Potential Risks or Red Flags**  
- [Each item should appear on its own line starting with `-`]

**Recommended Action Items**  
- [Each item should appear on its own line starting with `-`]

---

Here is the document content:
{extracted_text[:8000]}
"""
# Add the custom prompt (if any)
final_prompt = input.customPrompt + "\n\n" + base_prompt if input.customPrompt else base_prompt

response = call_groq(final_prompt)

    return {"insights": response}
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
