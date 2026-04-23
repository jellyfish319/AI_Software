import argparse
import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import ClapModel, ClapProcessor
from peft import PeftModel
from datasets import load_dataset, Audio
import librosa

MODEL_ID = "laion/clap-htsat-unfused"
LORA_CHECKPOINT = "./lora_checkpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def setup_model():
    print(f"[Info] Loading Model on {DEVICE}...")
    try:
        base_model = ClapModel.from_pretrained(MODEL_ID)
        processor = ClapProcessor.from_pretrained(MODEL_ID)
        
        if not os.path.exists(LORA_CHECKPOINT):
            print(f"[Warning] '{LORA_CHECKPOINT}' 경로를 찾을 수 없습니다. 기본 모델로 진행합니다.")
            return base_model.to(DEVICE), processor
            
        model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
        model.to(DEVICE)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"[Error] 모델 로드 실패: {e}")
        sys.exit(1)

def get_text_embeddings(model, processor):
    prompts = [f"This is a sound of {label.replace('_', ' ')}." for label in CLASS_NAMES]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        if isinstance(model, PeftModel):
            base = model.base_model
        else:
            base = model
        embeds = base.get_text_features(**text_inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds

def predict_audio(model, processor, audio_array, text_embeds, sr=48000):
    if len(audio_array) == 0: return -1, 0.0
    
    inputs = processor(
        audio=audio_array,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=sr * 4
    ).to(DEVICE)
    
    with torch.no_grad():
        if isinstance(model, PeftModel):
            audio_embeds = model.base_model.model.get_audio_features(input_features=inputs.input_features)
        else:
            audio_embeds = model.get_audio_features(**inputs)
            
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
        logits = audio_embeds @ text_embeds.t()
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = torch.max(logits.softmax(dim=-1)).item()
        
    return pred_idx, confidence

def print_evaluation_metrics(true_labels_idx, pred_labels_idx, mode_name):
    print("\n" + "="*50)
    print(f"FINAL EVALUATION RESULTS ({mode_name})")
    print("="*50)

    if not true_labels_idx:
        print("[Warning] 평가할 정답 데이터가 없습니다.")
        return

    acc = accuracy_score(true_labels_idx, pred_labels_idx)
    print(f"Overall Accuracy: {acc:.4f}")
    print("-" * 50)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(true_labels_idx, pred_labels_idx, average='weighted', zero_division=0)
    print("[ Weighted Average Metrics ]")
    print(f"  - Precision : {precision_w:.4f}")
    print(f"  - Recall    : {recall_w:.4f}")
    print(f"  - F1 Score  : {f1_w:.4f}")
    print("-" * 50)

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(true_labels_idx, pred_labels_idx, average=None, labels=range(len(CLASS_NAMES)), zero_division=0)
    
    class_metrics_df = pd.DataFrame({
        "Class Name": CLASS_NAMES,
        "Precision": precision_c,
        "Recall": recall_c,
        "F1 Score": f1_c,
        "Support": support_c
    })
    class_metrics_df = class_metrics_df.round(4)

    print("[ Per-class Metrics ]")
    print(class_metrics_df.to_string(index=False))
    print("-" * 50)
    
    # 4. Confusion Matrix 이미지 저장
    cm = confusion_matrix(true_labels_idx, pred_labels_idx, labels=range(len(CLASS_NAMES)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.title(f'Confusion Matrix ({mode_name}, Acc: {acc:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = f"{mode_name.lower()}_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix Image Saved: {save_path}")

def run_urban_evaluation(model, processor, text_embeds):
    print("\n" + "="*50)
    print(" [Mode: Urban] UrbanSound8K Test Set")
    print("="*50)
    
    CACHE_DIR = "./test/urban"
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    try:
        print(f"[Info] Loading UrbanSound8K dataset (Cache: {CACHE_DIR})...")
        dataset = load_dataset("danavery/urbansound8K", split="train", cache_dir=CACHE_DIR)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=48000))
        
        test_ds = dataset.filter(lambda x: x['fold'] == 10)
        print(f"[Info] Test Samples (Fold 10): {len(test_ds)}")
        
        true_labels_idx = []
        pred_labels_idx = []
        
        print("[Info] Running Inference...")
        for item in tqdm(test_ds):
            audio = item["audio"]["array"]
            label_idx = item["classID"]
            
            if len(audio) == 0: continue
            
            pred_idx, _ = predict_audio(model, processor, audio, text_embeds)
            
            true_labels_idx.append(label_idx)
            pred_labels_idx.append(pred_idx)

        print_evaluation_metrics(true_labels_idx, pred_labels_idx, "Urban_Fold10")
        
    except Exception as e:
        print(f"[Error] Urban 모드 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def run_custom_evaluation(model, processor, text_embeds):
    CUSTOM_DIR = "./test/custom"
    print("\n" + "="*50)
    print(f" [Mode: Custom] Evaluating files in '{CUSTOM_DIR}'")
    print("="*50)
    
    if not os.path.exists(CUSTOM_DIR):
        print(f"[Error] '{CUSTOM_DIR}' 폴더가 없습니다.")
        return

    audio_extensions = ('*.wav', '*.mp3', '*.flac')
    files = []
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(CUSTOM_DIR, "**", ext), recursive=True))
            
    if not files:
        print(f"[Warning] '{CUSTOM_DIR}' 폴더 및 하위 폴더에 오디오 파일이 없습니다.")
        return

    print(f"[Info] Found {len(files)} audio files. Running inference...")
    
    detailed_results = []
    true_labels_idx = []
    pred_labels_idx = []
    
    for file_path in tqdm(files):
        try:

            parent_dir = os.path.basename(os.path.dirname(file_path))
            ground_truth = parent_dir if parent_dir in CLASS_TO_IDX else None

            audio, _ = librosa.load(file_path, sr=48000)
            pred_idx, conf = predict_audio(model, processor, audio, text_embeds)
            pred_label = CLASS_NAMES[pred_idx]

            detailed_results.append({
                "Filename": os.path.relpath(file_path, CUSTOM_DIR),
                "Ground Truth": ground_truth if ground_truth else "(Unknown)",
                "Prediction": pred_label,
                "Confidence": f"{conf:.4f}"
            })

            if ground_truth is not None:
                true_labels_idx.append(CLASS_TO_IDX[ground_truth])
                pred_labels_idx.append(pred_idx)
                
        except Exception as e:
            print(f"[Error] processing {os.path.basename(file_path)}: {e}")

    df = pd.DataFrame(detailed_results)
    print("\n--- Individual Prediction Results ---")
    print(df.to_string(index=False))
    df.to_csv("custom_results.csv", index=False)
    print(f"[Info] Individual results saved to 'custom_results.csv'")

    if true_labels_idx:
        print(f"\n[Info] Calculating metrics for {len(true_labels_idx)} labeled files...")
        print_evaluation_metrics(true_labels_idx, pred_labels_idx, "Custom_Data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['urban', 'custom'],
                        help="Choose evaluation mode: 'urban' or 'custom'")
    args = parser.parse_args()
    
    model, processor = setup_model()
    text_embeds = get_text_embeddings(model, processor)
    
    if args.mode == "urban":
        run_urban_evaluation(model, processor, text_embeds)
    elif args.mode == "custom":
        run_custom_evaluation(model, processor, text_embeds)