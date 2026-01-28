import pickle
import torch
import re

# Load Vocab
with open("d:/final_year_project/final_year_project/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

print(f"Vocab size: {len(vocab)}")

sample_text = "I loved the practical labs."
print(f"Sample: '{sample_text}'")

# API Method (Split)
api_tokens = sample_text.lower().split()
api_indices = [vocab.get(w, 0) for w in api_tokens]
print(f"API Tokenization (split): {api_tokens}")
print(f"API Indices: {api_indices}")

# Correct Method (Regex matches CountVectorizer default)
regex_tokens = re.findall(r"(?u)\b\w\w+\b", sample_text.lower())
regex_indices = [vocab.get(w, 0) for w in regex_tokens]
print(f"Correct Tokenization (regex): {regex_tokens}")
print(f"Correct Indices: {regex_indices}")

# Check overlap
print(f"Index Difference: {set(api_indices) - set(regex_indices)}")
