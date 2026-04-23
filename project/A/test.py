import json
import torch
import os
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    pipeline,
    BlipProcessor, 
    BlipForConditionalGeneration
)

INPUT_FILE = "./data/test.json"
OUTPUT_DIR = "./release"
OUTPUT_FILENAME = "202001156.test.json"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

DB_PATH = "./chroma_db_final"
EMBEDDING_MODEL_PATH = "./models/movie_embedding_finetuned"
LLM_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
VISION_MODEL_ID = "Salesforce/blip-image-captioning-base"

def decode_base64_image(b64_string):
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception:
        return None

def load_system():
    print("Loading system models...", flush=True)

    if os.path.exists(EMBEDDING_MODEL_PATH):
        emb_path = EMBEDDING_MODEL_PATH
    else:
        emb_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_path,
        model_kwargs={'device': 'cuda'}
    )
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    vision_processor = BlipProcessor.from_pretrained(VISION_MODEL_ID)
    vision_model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL_ID).to("cuda")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    
    text_generator = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    )
    
    print("System loading complete!")
    return vectorstore, vision_processor, vision_model, tokenizer, text_generator

def generate_answer(question, vectorstore, v_proc, v_model, tokenizer, generator):
    docs = vectorstore.similarity_search(question, k=1)
    
    if docs:
        doc = docs[0]
        title = doc.metadata.get('title', 'Unknown')
        genres = doc.metadata.get('genres', 'Unknown')
        overview = doc.page_content
        b64_image = doc.metadata.get('image_data', '')

        visual_desc = "No image available."
        if b64_image:
            image = decode_base64_image(b64_image)
            if image:
                inputs = v_proc(images=image, text="A movie poster of", return_tensors="pt").to("cuda")
                out = v_model.generate(**inputs, max_new_tokens=50)
                caption = v_proc.decode(out[0], skip_special_tokens=True)
                visual_desc = caption

        context = f"Movie: {title}\nGenres: {genres}\nPlot: {overview}\nVisuals: {visual_desc}"
        
        prompt_messages = [
            {"role": "system", "content": "You are a movie QA bot. Answer based on the Context. Be concise and direct. If asked for genres, list them. If asked for a summary, provide one sentence."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
        
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        try:
            sequences = generator(prompt)
            answer = sequences[0]['generated_text'].strip()
            return answer
        except Exception as e:
            return f"Error: {str(e)}"

    return "No information found."

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    vectorstore, v_proc, v_model, tokenizer, generator = load_system()

    print(f"Reading input file: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    results = []
    print(f"Generating answers for {len(data)} questions...")
    
    for item in tqdm(data, desc="Processing"):
        question = item.get("question")
        if question:
            answer = generate_answer(question, vectorstore, v_proc, v_model, tokenizer, generator)
            results.append({
                "question": question,
                "answer": answer
            })

    print(f"Saving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("File generated successfully.")

if __name__ == "__main__":
    main()