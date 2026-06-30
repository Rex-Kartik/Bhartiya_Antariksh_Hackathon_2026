import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model, ModelConfig

def export_for_cpu(input_ckpt="checkpoints/best_model.pth", output_ckpt="checkpoints/model_cpu_quantized.pth"):
    print(f"Loading {input_ckpt}...")
    
    cfg = ModelConfig(checkpoint_path=input_ckpt, device="cpu")
    model = build_model(cfg)
    model.eval()

    print("Applying dynamic quantization to reduce size and improve CPU inference...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print(f"Saving stripped, CPU-optimized model to {output_ckpt}...")
    torch.save({"model": quantized_model.state_dict()}, output_ckpt)
    print("Done! You can now deploy this lightweight .pth file.")

if __name__ == "__main__":
    export_for_cpu()