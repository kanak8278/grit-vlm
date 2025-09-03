"""
Fine-tune SmolVLM on OK-VQA dataset using GRIT-KFAC.

This script demonstrates how to:
1. Load OK-VQA dataset 
2. Preprocess image-text pairs for SmolVLM
3. Apply GRIT adaptation with optimal device config
4. Train with Fisher Information updates
"""

import torch
from torch.utils.data import DataLoader
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized pixel values."""
    # Separate the different components
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Handle pixel values - just return as list since they have different shapes
    pixel_values = [item["pixel_values"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values
    }

from grit_vlm import GRITLoRAConfig
from grit_vlm.models.vlm_adapter import VLMGRITAdapter
from grit_vlm.config import get_device_config

class OKVQADataset(torch.utils.data.Dataset):
    """OK-VQA dataset for VQA training."""
    
    def __init__(self, dataset, processor, max_length=1200):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image and question
        image = item['image']
        question = item['question']
        answers = item['answers']  # List of possible answers
        
        # Use first answer as target (OK-VQA has multiple valid answers)
        target_answer = answers[0] if answers else "unknown"
        
        # Format for SmolVLM with image placeholder
        text_input = f"<image>Question: {question}\nAnswer:"
        # For labels, we don't need the image token
        text_target = f"Question: {question}\nAnswer: {target_answer}"
        
        # Process inputs
        inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Process targets for labels
        targets = self.processor(
            text=text_target,
            return_tensors="pt", 
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0), 
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }

def train_epoch(model, adapter, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update Fisher Information (GRIT's key feature)
        adapter.update_mixed_modal_fisher()
        
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Quick test with small dataset
        if batch_idx >= 50:  # Limit to 50 batches for testing
            break
    
    return total_loss / min(len(dataloader), 51)

def main():
    print("🚀 Fine-tuning SmolVLM on OK-VQA with GRIT-KFAC")
    print("=" * 50)
    
    # 1. Setup device configuration
    print("📱 Setting up device configuration...")
    device_kwargs = get_device_config("stable_training")
    
    # Get actual device used by the model
    if device_kwargs["device_map"] == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_kwargs["device_map"])
    
    print(f"✓ Device config: {device_kwargs['device_map']} -> actual device: {device}")
    
    # 2. Load SmolVLM model and processor
    print("\n📥 Loading SmolVLM model...")
    model = Idefics3ForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        **device_kwargs
    )
    
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    
    # 3. Apply GRIT adaptation
    print("\n🧩 Applying GRIT adaptation...")
    grit_config = GRITLoRAConfig(r=8, lora_alpha=16)  # Slightly larger for VQA
    adapter = VLMGRITAdapter(
        model=model,
        config=grit_config, 
        model_config_name="smolvlm_fast"  # Use fast config for testing
    )
    print(f"✓ GRIT applied to {len(adapter.grit_layers)} layers")
    
    # 4. Load OK-VQA dataset (small subset for testing)
    print("\n📚 Loading OK-VQA dataset...")
    dataset = load_dataset("lmms-lab/OK-VQA", split="val2014")
    
    # Use small subset for testing
    small_dataset = dataset.select(range(min(200, len(dataset))))
    print(f"✓ Using {len(small_dataset)} samples for training")
    
    # 5. Create dataset and dataloader
    train_dataset = OKVQADataset(small_dataset, processor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Use batch size 1 to avoid tensor shape issues
        shuffle=True
    )
    
    # 6. Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 7. Training loop
    print(f"\n🏋️ Starting training...")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: 1")
    print(f"GRIT layers: {len(adapter.grit_layers)}")
    
    try:
        avg_loss = train_epoch(model, adapter, train_loader, optimizer, device)
        print(f"\n✅ Training completed!")
        print(f"Average loss: {avg_loss:.4f}")
        
        # 8. Test inference
        print(f"\n🧪 Testing inference...")
        model.eval()
        
        # Get a sample from dataset
        sample = train_dataset[0] 
        
        # Test inference
        with torch.no_grad():
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                max_new_tokens=20,
                do_sample=False
            )
            
            # Decode response
            response = processor.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
            print(f"✓ Generated response: {response[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ GRIT-KFAC VQA TRAINING WORKS!")
        print("\n🎯 What was demonstrated:")
        print("  ✓ OK-VQA dataset loading and preprocessing") 
        print("  ✓ SmolVLM + GRIT integration")
        print("  ✓ Mixed-modal Fisher Information updates")
        print("  ✓ VQA-specific training loop")
        print("  ✓ Inference with fine-tuned model")
        print("\n🚀 Ready to scale to full dataset!")
    else:
        print("\n❌ Training setup needs debugging")