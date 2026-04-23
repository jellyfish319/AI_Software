import streamlit as st
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

# 페이지 설정 (가장 먼저 실행되어야 함)
st.set_page_config(page_title="Movie RAG Chatbot", layout="wide")

# 경로 설정
DB_PATH = "./chroma_db_final"
EMBEDDING_MODEL_PATH = "./models/movie_embedding_finetuned"
LLM_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
VISION_MODEL_ID = "Salesforce/blip-image-captioning-base"

@st.cache_resource
def load_models():
    """모델과 DB를 캐싱하여 로드 (새로고침 시 재로딩 방지)"""
    print("Loading models...", flush=True)
    
    # 1. Embedding
    if os.path.exists(EMBEDDING_MODEL_PATH):
        emb_path = EMBEDDING_MODEL_PATH
    else:
        emb_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_path,
        model_kwargs={'device': 'cuda'}
    )
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

    # 2. Vision (BLIP)
    vision_processor = BlipProcessor.from_pretrained(VISION_MODEL_ID)
    vision_model = BlipForConditionalGeneration.from_pretrained(VISION_MODEL_ID).to("cuda")

    # 3. LLM (Llama-3)
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
    
    return vectorstore, vision_processor, vision_model, tokenizer, text_generator

def decode_base64_image(b64_string):
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception:
        return None

def main():
    st.title("🎬 Multimodal Movie Chatbot")

    # 모델 로드 (최초 1회만 실행됨)
    with st.spinner("Loading AI Models... (This may take a minute)"):
        vectorstore, v_proc, v_model, tokenizer, generator = load_models()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_poster" not in st.session_state:
        st.session_state.current_poster = None
    if "current_movie_info" not in st.session_state:
        st.session_state.current_movie_info = {}

    # 사이드바: 영화 정보 및 포스터 표시
    with st.sidebar:
        st.header("Movie Context")
        if st.session_state.current_poster:
            st.image(st.session_state.current_poster, caption="Retrieved Poster", use_container_width=True)
        
        if st.session_state.current_movie_info:
            st.subheader(st.session_state.current_movie_info.get('title', ''))
            st.write(f"**Genres:** {st.session_state.current_movie_info.get('genres', '')}")
            with st.expander("Show Plot"):
                st.write(st.session_state.current_movie_info.get('overview', ''))
        else:
            st.info("Ask a question to find a movie!")

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("Ask about a movie..."):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching & Thinking..."):
                # 1. 검색 (Top-1)
                docs = vectorstore.similarity_search(prompt, k=1)
                
                context_text = "No movie data found."
                visual_description = "No image available."
                
                if docs:
                    doc = docs[0]
                    title = doc.metadata.get('title', 'Unknown')
                    genres = doc.metadata.get('genres', 'Unknown')
                    overview = doc.page_content
                    b64_image = doc.metadata.get('image_data', '')

                    # 사이드바 업데이트를 위한 정보 저장
                    st.session_state.current_movie_info = {
                        'title': title,
                        'genres': genres,
                        'overview': overview
                    }

                    # 2. 이미지 분석
                    if b64_image:
                        image = decode_base64_image(b64_image)
                        st.session_state.current_poster = image # 사이드바 이미지 업데이트
                        
                        if image:
                            inputs = v_proc(images=image, text="A movie poster of", return_tensors="pt").to("cuda")
                            out = v_model.generate(**inputs, max_new_tokens=50)
                            caption = v_proc.decode(out[0], skip_special_tokens=True)
                            visual_description = caption
                    else:
                        st.session_state.current_poster = None

                    # 3. 문맥 조립
                    context_text = f"""
                    Movie Title: {title}
                    Genres: {genres}
                    Plot Overview: {overview}
                    Visual Analysis of Poster: {visual_description}
                    """
                
                # 4. LLM 추론
                prompt_template = [
                    {"role": "system", "content": "You are a movie expert. Answer based on the Context. Use 'Visual Analysis' to describe the poster."},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question:\n{prompt}"}
                ]
                
                formatted_prompt = tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=True)
                sequences = generator(formatted_prompt)
                full_response = sequences[0]['generated_text'].strip()

            # 응답 표시 및 저장
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # 사이드바 강제 업데이트 (이미지 표시를 위해)
            st.rerun()

if __name__ == "__main__":
    main()