
import json
import os

def create_notebook():
    cells = []
    
    # Title and Intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# DataLFlash Demo: 3 Steps to Massive Dataset Loading ⚡\n",
            "\n",
            "This notebook demonstrates the 3-step process for using DataLFlash with massive datasets (Option A in the README).\n",
            "\n",
            "**The 3 Steps:**\n",
            "1. **Convert Dataset**: Convert your standard PyTorch dataset into optimized chunks.\n",
            "2. **Load Dataloaders**: Automatically load train/val/test loaders.\n",
            "3. **Train**: Use the loader just like a standard PyTorch DataLoader."
        ]
    })
    
    # Step 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Convert Dataset 📦\n",
            "\n",
            "First, we'll create a dummy PyTorch dataset and then convert it into DataLFlash chunks."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from torch.utils.data import TensorDataset\n",
            "from datalflash.utils import DatasetConverter\n",
            "import shutil\n",
            "import os\n",
            "\n",
            "# 1. Create a dummy dataset (Simulating a massive dataset)\n",
            "print(\"Creating dummy dataset...\")\n",
            "X = torch.randn(1000, 3, 32, 32) # 1000 images of 32x32\n",
            "y = torch.randint(0, 10, (1000,)) # 1000 labels\n",
            "my_dataset = TensorDataset(X, y)\n",
            "\n",
            "# Define output directory\n",
            "output_dir = \"./demo_data_chunks\"\n",
            "\n",
            "# Clean up previous run if exists\n",
            "if os.path.exists(output_dir):\n",
            "    shutil.rmtree(output_dir)\n",
            "\n",
            "# 2. Convert to DataLFlash Chunks\n",
            "print(\"Converting to chunks...\")\n",
            "DatasetConverter.create_chunked_dataset(\n",
            "    pytorch_dataset=my_dataset,\n",
            "    output_dir=output_dir,\n",
            "    chunk_size=100, # Small chunk size for this demo\n",
            "    split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1},\n",
            "    shuffle=True\n",
            ")"
        ]
    })
    
    # Step 2
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Load Dataloaders 🚀\n",
            "\n",
            "Now we load the dataloaders from the chunked directory. DataLFlash handles the background loading."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from datalflash.core import get_dataloaders\n",
            "\n",
            "print(\"Loading dataloaders...\")\n",
            "loaders = get_dataloaders(\n",
            "    chunks_dir=output_dir,\n",
            "    batch_size=32\n",
            ")\n",
            "\n",
            "train_loader = loaders['train']\n",
            "val_loader = loaders['val']\n",
            "test_loader = loaders['test']\n",
            "\n",
            "print(f\"Train batches: {len(train_loader)}\")\n",
            "print(f\"Val batches: {len(val_loader)}\")\n",
            "print(f\"Test batches: {len(test_loader)}\")"
        ]
    })
    
    # Step 3
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Train 🔥\n",
            "\n",
            "Finally, we iterate through the loader just like a standard PyTorch DataLoader."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Starting training loop...\")\n",
            "for i, (features, targets) in enumerate(train_loader):\n",
            "    # Your training logic here\n",
            "    if i == 0:\n",
            "        print(f\"Batch {i}: Features {features.shape}, Targets {targets.shape}\")\n",
            "    \n",
            "    # Simulate training\n",
            "    pass\n",
            "\n",
            "print(\"Training loop finished!\")"
        ]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open("demo_chunking.ipynb", "w", encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        
    print("Notebook 'demo_chunking.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
