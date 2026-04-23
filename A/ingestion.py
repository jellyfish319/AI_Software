import os
import shutil
import base64
from io import BytesIO
from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATASET_NAME = "stzhao/movie_posters_100k_controlnet"
DB_PATH = "./chroma_db_final"
MODEL_PATH = "./models/movie_embedding_finetuned"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500
LIMIT_DATA = 100000

def main():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Existing DB ({DB_PATH}) deleted. Creating new DB.")

    if os.path.exists(MODEL_PATH):
        print(f"Using fine-tuned model: {MODEL_PATH}")
        emb_model_name = MODEL_PATH
    else:
        print(f"Fine-tuned model not found. Using base model: {BASE_MODEL}")
        emb_model_name = BASE_MODEL

    print("Loading Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_model_name,
        model_kwargs={'device': 'cuda'}
    )

    print("Loading dataset (Streaming mode)...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    print(f"Starting Vector DB construction (Max {LIMIT_DATA})...")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    docs_buffer = []
    saved_count = 0
    
    for i, item in enumerate(dataset):
        if i >= LIMIT_DATA: break

        title = item.get("title", "Unknown Movie")

        raw_content = item.get("caption_with_overview", "")
        if not raw_content:
            raw_content = item.get("overview", item.get("caption", ""))
        
        if not raw_content or len(raw_content) < 5: 
            continue

        final_page_content = f"Title: {title}. Overview: {raw_content}"

        overview = item.get("overview", "")
        item_id = item.get("id", i)
        
        genres_list = item.get("genres", [])
        genres_str = ""
        if isinstance(genres_list, list):
            genres_names = [g.get('name', '') for g in genres_list if isinstance(g, dict)]
            genres_str = ", ".join(filter(None, genres_names))

        image_obj = item.get("image")
        b64_string = ""
        
        if image_obj:
            try:
                buffered = BytesIO()
                image_obj.save(buffered, format="JPEG") 
                b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Image conversion failed ({title}): {e}")

        doc = Document(
            page_content=final_page_content,
            metadata={
                "id": item_id,
                "title": title,
                "genres": genres_str,
                "overview": overview,
                "image_data": b64_string,
                "source": "poster_dataset"
            }
        )
        docs_buffer.append(doc)

        if len(docs_buffer) >= BATCH_SIZE:
            vectorstore.add_documents(docs_buffer)
            saved_count += len(docs_buffer)
            docs_buffer = []
            print(f"Saved {saved_count} documents...")

    if docs_buffer:
        vectorstore.add_documents(docs_buffer)
        saved_count += len(docs_buffer)

    print(f"DB construction complete. Total {saved_count} documents saved.")
    print(f"Save path: {DB_PATH}")

if __name__ == "__main__":
    main()