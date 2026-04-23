
import os
try:
    import peft
    import soundfile
except ImportError:
    print("Installing dependencies...")
    os.system("pip install -q datasets[audio] transformers peft soundfile librosa scikit-learn")
    print("Installation Complete! Please restart the runtime.")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import ClapModel, ClapProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "laion/clap-htsat-unfused"

print(f"Using Device: {DEVICE}")

print("Loading and Resampling Dataset...")
dataset = load_dataset("danavery/urbansound8K", split="train")

dataset = dataset.cast_column("audio", Audio(sampling_rate=48000))

train_ds = dataset.filter(lambda x: x['fold'] != 10)
test_ds = dataset.filter(lambda x: x['fold'] == 10)

print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

print("Preparing Model and Text Embeddings...")
original_model = ClapModel.from_pretrained(MODEL_ID)
processor = ClapProcessor.from_pretrained(MODEL_ID)
original_model.to(DEVICE)

class_names = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]
prompts = [f"This is a sound of {label.replace('_', ' ')}." for label in class_names]

text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
with torch.no_grad():
    fixed_text_embeds = original_model.get_text_features(**text_inputs)
    fixed_text_embeds = fixed_text_embeds / fixed_text_embeds.norm(dim=-1, keepdim=True)

print("Setting up LoRA...")
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "dense"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(original_model, config)
model.print_trainable_parameters()
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

def collate_fn(batch):
    audios = [item["audio"]["array"] for item in batch]
    labels = [item["classID"] for item in batch]

    inputs = processor(
        audio=audios,
        sampling_rate=48000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=48000*4 
    )

    return inputs.input_features, torch.tensor(labels)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)


print("\nStarting LoRA Training...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for audio_features, labels in progress_bar:
        audio_features = audio_features.to(DEVICE)
        labels = labels.to(DEVICE)

        audio_embeds = model.base_model.model.get_audio_features(input_features=audio_features)
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

        logits = audio_embeds @ fixed_text_embeds.t()

        try:
            logit_scale = model.base_model.model.logit_scale.exp()
        except AttributeError:

            logit_scale = 14.3

        logits = logits * logit_scale

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")


print("\nRunning Evaluation on Test Set...")
model.eval()

true_labels = []
pred_labels = []

with torch.no_grad():
    for audio_features, labels in tqdm(test_loader):
        audio_features = audio_features.to(DEVICE)

        audio_embeds = model.base_model.model.get_audio_features(input_features=audio_features)
        audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)

        logits = audio_embeds @ fixed_text_embeds.t()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        true_labels.extend(labels.numpy())
        pred_labels.extend(preds)

acc = accuracy_score(true_labels, pred_labels)
print(f"\n[Final Result] LoRA Fine-tuned Accuracy: {acc:.4f}")

save_directory = "./lora_checkpoint"
model.save_pretrained(save_directory)

print(f"LoRA weights saved in '{save_directory}'")