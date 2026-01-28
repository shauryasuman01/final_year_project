import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import data_processor
from quantum_model import QNLPHybridModel
import os

# Hyperparameters
VOCAB_SIZE = 500  # Limiting vocab for simplicity in QNLP
EMBED_DIM = 8     # Must match what the QLSTM expects (via linear reduction inside)
HIDDEN_SIZE = 4   # Qubits
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.01

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=10):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Simple tokenization: split by space and map to indices
        tokens = text.lower().split()
        indices = [self.tokenizer.get(w, 0) for w in tokens[:self.max_len]]
        
        # Padding
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def build_vocab(texts, vocab_size):
    vectorizer = CountVectorizer(max_features=vocab_size)
    vectorizer.fit(texts)
    vocab = vectorizer.vocabulary_
    # Add padding token and UNK
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    # Shift other indices
    for k in vocab:
        if k not in ['<PAD>', '<UNK>']:
            vocab[k] += 2
    return vocab

def main():
    print("Loading Data...")
    if os.path.exists("d:/final_year_project/final_year_project/processed_feedback.csv"):
        df = pd.read_csv("d:/final_year_project/final_year_project/processed_feedback.csv")
    else:
        df = data_processor.load_and_process_data('d:/final_year_project/final_year_project/student_feedback.csv')

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['combined_text'].values, df['sentiment_label'].values, test_size=0.2, random_state=42
    )

    print("Building Vocabulary...")
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    
    train_dataset = FeedbackDataset(train_texts, train_labels, vocab)
    test_dataset = FeedbackDataset(test_texts, test_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("Initializing Quantum Model...")
    model = QNLPHybridModel(vocab_size=len(vocab)+2, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.4f}")

    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
    print(f"Test Accuracy: {test_correct/test_total:.4f}")
    
    # Save Model
    torch.save(model.state_dict(), "d:/final_year_project/final_year_project/qnlp_model.pth")
    print("Model saved to qnlp_model.pth")
    
    # Save Vocab
    import pickle
    with open("d:/final_year_project/final_year_project/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to vocab.pkl")

if __name__ == "__main__":
    main()
