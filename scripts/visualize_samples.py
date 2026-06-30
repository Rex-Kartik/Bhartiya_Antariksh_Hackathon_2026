import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Exact training parameters extracted from preprocess.py
PREPROCESS = {
    'bt_min': 180.0, 
    'bt_max': 320.0
}

def denormalize(arr):
    """Converts [0.0, 1.0] tensor array back to real Kelvin temperatures."""
    return (arr * (PREPROCESS['bt_max'] - PREPROCESS['bt_min'])) + PREPROCESS['bt_min']

def plot_sample(npz_path):
    data = np.load(npz_path)
    
    # Extract base input frames
    f0 = data["f0"]
    f1 = data["f1"]

    # 1. Reconstruct the NaN mask from the inputs.
    # In preprocess.py, deep space NaNs were forced to 0.0 before training.
    # We find those exact pixels to revert them back to NaNs.
    nan_mask = (f0 == 0.0) | (f1 == 0.0)
    
    # Group frames for easy iteration
    frames_data = [
        ("Frame 0", f0),
        ("Pred T0.25", data["t025"]),
        ("Pred T0.50", data["t050"]),
        ("Pred T0.75", data["t075"]),
        ("Frame 1", f1),
        ("GT T0.50", data["gt_050"])
    ]
    
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    for i, (title, img) in enumerate(frames_data):
        # 2. Denormalize back to Kelvin
        img_denorm = denormalize(img.copy())
        
        # 3. Re-apply NaN mask (revert 0.0 -> np.nan)
        # Matplotlib automatically renders np.nan as transparent/background color
        img_denorm[nan_mask] = np.nan
        
        # 4. Plot with strictly calibrated temperature bounds
        im = axes[i].imshow(
            img_denorm, 
            cmap="inferno", 
            vmin=PREPROCESS['bt_min'], 
            vmax=PREPROCESS['bt_max']
        )
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].axis("off")
    
    # Add a shared colorbar at the end to show actual temperatures
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.04)
    cbar.set_label("Brightness Temperature (Kelvin)", rotation=270, labelpad=15)
    
    plt.suptitle(f"Interpolation Results: {Path(npz_path).name}", fontsize=16, y=1.05)
    plt.show()

if __name__ == "__main__":
    # Update this path to the sample file you want to visualize
    sample_path = Path("logs/samples/step_0000500.npz") 
    
    if sample_path.exists():
        plot_sample(sample_path)
    else:
        print(f"Sample file not found at {sample_path}")
        print("Please check your logs/samples/ directory for available .npz files.")