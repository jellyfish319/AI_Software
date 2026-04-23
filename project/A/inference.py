import torch
import os
import base64
from io import BytesIO
from PIL import Image
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

DB_PATH = "./chroma_db_final"
EMBEDDING_MODEL_PATH = "./models/movie_embedding_finetuned"
LLM_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
VISION_MODEL_ID = "Salesforce/blip-image-captioning-base"

def decode_base64_image(b64_string):
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Image decode error: {e}")
        return None

def main():
    print("Loading Multimodal RAG Chatbot...", flush=True)

    print("Loading Embedding Model...")
    if os.path.exists(EMBEDDING_MODEL_PATH):
        model_path = EMBEDDING_MODEL_PATH
    else:
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'}
    )
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    print("Vector DB Connected.")

    print(f"Loading Vision Model ({VISION_MODEL_ID})...")
    vision_processor = BlipProcessor.from_pretrained(VISION_MODEL_ID)
    vision_model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL_ID).to("cuda")

    print("Loading LLM (Llama-3) in 4-bit...")
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
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        return_full_text=False
    )
    print("System Ready! (Text + Image Analysis Supported)\n")

    while True:
        query = input("\nQuestion (quit: q): ")
        if query.lower() in ["q", "quit", "exit"]:
            print("Bye!")
            break
        
        print(f"Analyzing Query: '{query}'")

        docs = vectorstore.similarity_search(query, k=1)
        
        context_text = ""
        visual_description = "No image available."
        
        if docs:
            doc = docs[0]
            title = doc.metadata.get('title', 'Unknown')
            genres = doc.metadata.get('genres', 'Unknown Genres')
            overview = doc.page_content
            b64_image = doc.metadata.get('image_data', '')
            
            print(f"Found Movie: {title}")
            print(f"Genres: {genres}")

            if b64_image:
                print("Analyzing Poster Image with Vision AI...", end="", flush=True)
                image = decode_base64_image(b64_image)
                
                if image:
                    inputs = vision_processor(images=image, text="A movie poster of", return_tensors="pt").to("cuda")
                    out = vision_model.generate(**inputs, max_new_tokens=50)
                    caption = vision_processor.decode(out[0], skip_special_tokens=True)
                    
                    visual_description = caption
                    print(f" Done!\nVision Insight: '{visual_description}'")
            
            context_text = f"""
            Movie Title: {title}
            Genres: {genres}
            Plot Overview: {overview}
            Visual Analysis of Poster: {visual_description}
            """
        else:
            print("No relevant movie found.")
            context_text = "No movie data found."

        prompt_template = [
            {"role": "system", "content": "You are a movie expert capable of analyzing both plot and visual elements. Use the 'Visual Analysis' to answer questions about the poster's style, time period, or mood."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question:\n{query}"}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True)

        print("Thinking...", end="", flush=True)
        sequences = text_generator(formatted_prompt)
        answer_en = sequences[0]['generated_text'].strip()

        print(f"\n\nAnswer:\n{answer_en}")

if __name__ == "__main__":
    main()