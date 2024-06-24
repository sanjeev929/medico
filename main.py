from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pdfplumber
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import pipeline as ner_pipeline

# Initialize FastAPI app
app = FastAPI()

# Example storage for uploaded content
uploaded_content: Dict[str, str] = {}

# Example training data for document classification (you need labeled data for actual training)
training_data = [
    ("Patient Name: John Doe Date of Birth: January 1, 1980 Medical History: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Diagnosis: Hypertension Medications: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Recommendations: Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque", "medical_report"),
    ("Patient Name: Jane Doe Date of Birth: February 2, 1990 No significant medical history. Diagnosis: Normal X-ray", "radiology_report"),
    ("Patient Name: Emily Johnson Date of Birth: April 4, 2000 No medical history. Recommendation: Follow-up in 6 months for routine check-up", "general_report")
]

# Train a simple classifier (example)
pipeline_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])

pipeline_classifier.fit([text for text, label in training_data], [label for text, label in training_data])

# Initialize NER pipeline from Hugging Face Transformers
ner = ner_pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Pydantic models for request/response validation
class DocumentType(BaseModel):
    filename: str

class ClassificationResponse(BaseModel):
    document_type: str

class NEREntity(BaseModel):
    text: str
    label: str

class NERResponse(BaseModel):
    entities: List[NEREntity]

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from image using pytesseract
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to classify document type based on text content
def classify_document(text):
    return pipeline_classifier.predict([text])[0]

# Route to upload files and classify their content
@app.post("/load", response_model=ClassificationResponse)
async def load_file(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file.file)
    else:
        text = extract_text_from_image(file.file)
    
    uploaded_content[file.filename] = text

    return {"document_type": classify_document(text)}

# Route to classify document type based on uploaded filename
@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(doc: DocumentType):
    filename = doc.filename
    if filename not in uploaded_content:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    
    text = uploaded_content[filename]
    document_type = classify_document(text)
    
    return {"document_type": document_type}

# Route to extract named entities from uploaded documents
@app.post("/query", response_model=NERResponse)
async def query_content(doc: DocumentType):
    filename = doc.filename
    if filename not in uploaded_content:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    
    text = uploaded_content[filename]
    entities = ner(text)
    
    formatted_entities = []
    for entity in entities:
        formatted_entities.append({
            "text": entity["word"],  # Actual text of the entity
            "label": entity["entity"]  # Label of the entity
        })
    
    return {"entities": formatted_entities}

# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
