from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import io
import pickle
import sys
import os

# Add project root to path for imports
sys.path.append("d:/final_year_project/final_year_project")

from quantum_model import QNLPHybridModel
from data_processor import load_and_process_data

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
VOCAB_PATH = "d:/final_year_project/final_year_project/vocab.pkl"
MODEL_PATH = "d:/final_year_project/final_year_project/qnlp_model.pth"

LABELS = {0: "Negative", 1: "Positive"}

model = None
vocab = None
MAX_LEN = 10

def load_resources():
    global model, vocab
    
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        print("Warning: Model or Vocab not found. Run training first.")
        return

    # Load Vocab
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print("Vocab loaded.")

    # Load Model
    model = QNLPHybridModel(vocab_size=len(vocab)+2, embed_dim=8, hidden_size=4)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded.")

# Load on startup
load_resources()

import re

# ...

def tokenize(text, vocab, max_len=10):
    # Use standard Regex tokenization matching CountVectorizer default: r"(?u)\b\w\w+\b"
    # This strips punctuation and ignores single characters
    tokens = re.findall(r"(?u)\b\w\w+\b", text.lower())
    indices = [vocab.get(w, 0) for w in tokens[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0) # Batch size 1

@app.post("/analyze")
async def analyze_feedback(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # We need to process the dataframe similarly to training
        # Identify feedback columns
        feedback_cols = [
            "What did you like most about this course and why?",
            "Which topics were most useful for your understanding or future career?",
            "Which topics were difficult or unclear? Please mention briefly.",
            "How effective was the teaching method used in this course?",
            "Were the lectures and study materials helpful? Explain shortly.",
            "How can this course be improved for future students?",
            "Did this course meet your expectations? Why or why not?",
            "How was the pace of teaching (too fast, slow, or balanced)? Explain briefly.",
            "What practical skills or knowledge did you gain from this course?",
            "Any other suggestions or comments?"
        ]
        
        # Check if columns exist, if not, try to find any text column or simple 'feedback'
        available_cols = [c for c in feedback_cols if c in df.columns]
        
        if not available_cols:
             # Fallback: looks for 'feedback' or use all string columns
             text_cols = df.select_dtypes(include=['object']).columns
             if len(text_cols) > 0:
                 df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
             else:
                 raise HTTPException(status_code=400, detail="No suitable text columns found for analysis.")
        else:
            df['combined_text'] = df[available_cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

        results = []
        positive_count = 0
        negative_count = 0
        
        for idx, row in df.iterrows():
            text = row['combined_text']
            inputs = tokenize(text, vocab)
            
            with torch.no_grad():
                output = model(inputs)
                prob = output.item()
                label = 1 if prob > 0.5 else 0
            
            res = {
                "id": idx,
                "name": row.get("Name", "Anonymous"),
                "text": text[:100] + "..." if len(text) > 100 else text, # Snippet
                "sentiment": LABELS[label],
                "score": round(prob, 4)
            }
            results.append(res)
            
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1
        
        return {
            "total_processed": len(results),
            "summary": {
                "positive": positive_count,
                "negative": negative_count
            },
            "detailed_results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "QNLP API Active"}
