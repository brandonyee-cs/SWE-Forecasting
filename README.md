# Snow Water Equivalent (SWE) Forecasting Model

## Overview
This repository contains a deep learning model for forecasting Snow Water Equivalent (SWE) using multisensor spatiotemporal data. The model combines Transformer, ConvLSTM, and U-Net architectures to accurately predict future SWE distributions based on historical observations.

## Project Structure
```
swe-forecasting/
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── run.py                # Main training script
└── src/
    ├── data_generation.py # Data handling and preprocessing
    └── models.py          # Neural network architecture definitions
```

## Model Architecture
The main model (`SWETransUNet`) integrates three powerful deep learning components:

1. **Transformer**: Processes patches of spatiotemporal data through self-attention mechanisms
2. **ConvLSTM**: Captures temporal dependencies in the data
3. **U-Net**: Handles the spatial upsampling for high-resolution predictions

The architecture can process multivariate sensor inputs (4 sensors) over 12 months to predict future SWE distributions.

### Key Components:
- **PatchEmbedding**: Converts input tensor into embedded patches using 3D convolution
- **TransformerEncoder**: Applies self-attention and feed-forward transformations
- **ConvLSTM**: Processes sequence data while preserving spatial relationships
- **UNetDecoderBlock**: Performs upsampling and convolution operations for reconstruction

## Data Format
The model expects data in the following format:
- **Input (X)**: 12 months of data from 4 sensors, shape: `[batch_size, 4, 12, 64, 64, 3]`
- **Target (y)**: 1 month of forecasted data, shape: `[batch_size, 64, 64, 3]`

Each sample represents a spatial grid of SWE measurements with 3 channels.

## Usage

### Installation
```bash
git clone https://github.com/IMSS-Lab/SWE-Forecasting.git
cd swe-forecasting
pip install -r requirements.txt
```

### Training
To train the model using default parameters:

```bash
python run.py
```

Make sure to update the `data_path` variable in `run.py` to point to your dataset.

### Customizing Training
You can customize the training process by modifying these parameters in `run.py`:

```python
# Model parameters
model_params = {
    'emb_dim': 384,
    'num_heads': 4,
    'hidden_dim': 512,
    'num_transformer_layers': 2,
    'patch_size': (2, 4, 4),
    'dropout': 0.1
}

# Training parameters
batch_size = 4
num_epochs = 5
learning_rate = 1e-5
weight_decay = 1e-4
early_stopping_patience = 3
```

## Data Augmentation
The project includes data augmentation techniques to improve model generalization:

- **RandomNoise**: Adds random noise to input data
- **RandomFlip**: Randomly flips input and target data horizontally or vertically

To enable data augmentation, use the `get_augmented_data()` function instead of `prepare_data()` in `run.py`.

## Alternative Models
The repository also includes `SWEUNet`, a simplified U-Net architecture without transformer components, which can be used as a baseline for comparison.

## Results
The training script automatically generates:
- Loss and MAE plots
- Sample predictions visualization
- Checkpoint files for model resumption

Results are saved in the specified output directory (default: `./swe_model_results`).

## Requirements
See `requirements.txt` for the complete list of dependencies.

## License
[Add your license information here]

## Citation
[Add citation information if applicable]

## Contact
[Your contact information]