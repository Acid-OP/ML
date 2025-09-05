# Project Setup and Testing Guide

## Initial Setup

### 1. Clone the Project
```bash
git clone https://github.com/Acid-OP/ML
```

### 2. Install UV Package Manager
- If `uv` is not installed, run:
  ```bash
  pip install uv
  ```
- If `pip` is also not available, download Python first:
  - Go to [python.org](https://python.org) and download the latest version
  - Install Python with pip included
- After installation, navigate to the cloned project directory and run:
  ```bash
  uv sync
  ```

### 3. Install CUDA 12.6
- Go to Google and search "download CUDA 12.6 version"
- Download the CUDA Toolkit 12.6
- Install it on your machine following the installation wizard

## Project Configuration

### 4. Open Project in VS Code
- Launch VS Code in the directory where you cloned the project

### 5. Activate Virtual Environment
```bash
source .venv/Scripts/activate
```

### 6. Install PyTorch with CUDA Support
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 7. Wait for Installation
- Allow all packages to install completely

## Testing and Verification

### 8. GPU Detection Test
edit any file and paste this code snippet to test GPU availability: (just for testing after that undo that edit)

```python
import torch
print("CUDA available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected by PyTorch")
```

**Expected Output:** The code should detect your GPU and run on GPU, not CPU.

### 9. Run Fine-tuning
Once GPU is verified, execute the fine-tuning script:
```bash
uv run python finetune.py
```
*(Or check the exact command if different - gpt it )*

### 10. Fix File Paths
- Update all file paths in the code to match your system setup (acc to your machine/pc)
- Replace hardcoded paths that were configured for my pc.

### 11. Run Test File
- Execute the test file after path corrections

### 12. Image Testing
- Currently configured for single image testing
- Change the image path to any image from the dataset or your own
- Verify the model correctly identifies:
  - "non" for non-mangrove images
  - "mangrove" for mangrove images

### 13. Batch Testing (Optional)
- Test with entire folders instead of single images
- This simulates how the final application will work
- Research the exact command needed for batch processing
### 14. Ignore the warnings that come while running the code just focus on the output that comes in the last

## TODO Items

### 1. Dataset Improvement
- **Priority:** Find better datasets for:
  - Seagrass classification
  - Saltmarsh identification  
  - Enhanced mangrove detection
- **Goal:** Improve overall model performance

### 2. Research Requirements
- **Land Requirements:** Research optimal land conditions for each crop type
- **Environmental Factors:** Study what these crops need to thrive
- **Validation Logic:** Add checks to verify if uploaded images/coordinates meet crop requirements

### 3. Performance Evaluation
- Test it and tell me how it is working (for me it worked and gave right answers)
---

**Note:** This setup is specifically configured for mangrove detection right now. The model has been fine-tuned on a small dataset but appears to be functioning correctly for initial testing purposes. Similarly I'll do for the rest of the crops with better dataset (if w eget any ) but first lets confirm this code by testing it .
