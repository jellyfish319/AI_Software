import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from datasets import load_dataset
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = "./models/movie_embedding_finetuned"
DATASET_NAME = "stzhao/movie_posters_100k_controlnet"
BATCH_SIZE = 32
EPOCHS = 1

def main():
    print(f"Embedding model training started (Base: {MODEL_NAME})")
    
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    train_examples = []
    limit = 100000
    
    print(f"Processing training data (Max {limit})...")
    
    for i, item in enumerate(dataset):
        if i >= limit: 
            break
            
        title = item.get('title', '')
        genres = item.get('genres', '')
        content = item.get('caption_with_overview', item.get('overview', item.get('caption', '')))
        
        if not title or not content or len(content) < 10:
            continue
            
        query_text = f"Movie title: {title}, Genre: {genres}"
        doc_text = f"Title: {title}. Overview: {content}"
        
        train_examples.append(InputExample(texts=[query_text, doc_text]))

    print(f"Training data ready: {len(train_examples)} pairs")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Training in progress...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )

    print(f"Training complete. Model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()