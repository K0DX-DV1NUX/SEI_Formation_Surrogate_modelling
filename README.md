# SEI Formation Surrogate Modelling

A deep learning-based surrogate model for predicting Solid Electrolyte Interface (SEI) formation in battery systems. This project implements and compares multiple neural network architectures (MLP, CNN, LSTM, GRU) to create efficient surrogate models that can predict battery behavior without expensive simulations.

Install requirements with:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── models/                      # Neural network model implementations
│   ├── MLP.py                  # Multi-Layer Perceptron
│   ├── CNN.py                  # Convolutional Neural Network
│   ├── LSTM.py                 # Long Short-Term Memory
│   └── GRU.py                  # Gated Recurrent Unit
├── builds/                      # Data preprocessing and building
│   ├── build_dataframes.py     # Load and structure data
│   ├── build_datasets.py       # Create PyTorch datasets
│   └── standardization.py      # Data normalization
├── scripts/                     # Training scripts for each model
│   ├── MLP.sh
│   ├── CNN.sh
│   ├── LSTM.sh
│   └── GRU.sh
├── utils/                       # Utility functions
│   └── utilities.py            # Plotting, evaluation, helpers
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Model predictions and metrics
├── plots/                       # Visualization outputs
├── Experiments/                 # Dataset directories
│   ├── train/                  # Training data
│   ├── vali/                   # Validation data
│   ├── test1/                   # Keep 1 Test data (Multi-stage)
│   ├── test2/                   # Keep 1 Test data (Fast)
│   └── test3/                   # keep 1 Test data (Slow)
├── exp.py                       # Main experiment class
├── run_exp.py                   # Entry point script
└── Dataset_generation.ipynb    # Notebook for data generation
```

## Usage

### Training a Model

Basic training with default parameters:
```bash
python run_exp.py --mode train --model MLP
```

Advanced training with custom hyperparameters:
```bash
python run_exp.py \
  --mode train \
  --model CNN \
  --epochs 150 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --window_size 30 \
  --seed 42 \
  --patience 10
```

### Testing a Trained Model

```bash
python run_exp.py \
  --mode test \
  --test_model_folder checkpoints/CNN_42 \
  --test_dir Experiments/test
```

### Using Shell Scripts

Pre-configured training scripts are available:
```bash
bash scripts/MLP.sh
bash scripts/CNN.sh
bash scripts/LSTM.sh
bash scripts/GRU.sh
```

## Command-Line Arguments

### Mode
- `--mode`: Run mode - `train` or `test` (default: `train`)

### Data Paths
- `--train_dir`: Path to training data (default: `Experiments/train`)
- `--vali_dir`: Path to validation data (default: `Experiments/vali`)
- `--test_dir`: Path to test data (default: `Experiments/test`)

### Output Directories
- `--plots_dir`: Directory for saving plots (default: `plots`)
- `--checkpoints_dir`: Directory for model checkpoints (default: `checkpoints`)
- `--results_dir`: Directory for results (default: `results`)

### Model Architecture
- `--model`: Model type - `MLP`, `CNN`, `LSTM`, or `GRU` (default: `MLP`)
- `--in_features`: Number of input features (default: `3`)
- `--out_features`: Number of output features (default: `2`)
- `--window_size`: Lookback window size for time series (default: `30`)

### Training Parameters
- `--epochs`: Number of training epochs (default: `100`)
- `--batch_size`: Batch size (default: `256`)
- `--learning_rate`: Initial learning rate (default: `0.001`)
- `--loss`: Loss function - `mse` or `mae` (default: `mse`)
- `--patience`: Early stopping patience (default: `5`)

### Other Options
- `--seed`: Random seed for reproducibility (default: `42`)
- `--num_workers`: Number of workers for data loading (default: `10`)
- `--device`: Device to use - `cuda` or `cpu` (auto-detected)

## Models

### MLP (Multi-Layer Perceptron)
Fully connected feedforward network suitable for sequence classification and tabular data.

### CNN (Convolutional Neural Network)
Extracts local temporal patterns using 1D convolutions with batch normalization.

### LSTM (Long Short-Term Memory)
Captures long-term dependencies in time series data with gating mechanisms.

### GRU (Gated Recurrent Unit)
Simplified RNN variant with fewer parameters than LSTM, suitable for sequence learning.

## Data Format

The project expects CSV files in the `Experiments` folder with the following structure:
- Training data: `Experiments/train/`
- Validation data: `Experiments/vali/`
- Test data: `Experiments/test/`

Each CSV file should contain time series data with features matching `--in_features` parameter.

## Output

After training or testing:
- **Checkpoints**: Saved in `checkpoints/{model}_{seed}/`
- **Results**: CSV files with predictions and targets in `results/{model}_{seed}/`
- **Plots**: Visualization plots in `plots/{model}_{seed}/`
- **Standardization**: Scaling parameters in `checkpoints/{model}_{seed}/std_values.json`

## Development

### Running Experiments with Dataset Generation

Use the included Jupyter notebook for data exploration and generation:
```bash
jupyter notebook Dataset_generation.ipynb
```

### Data Cleaning

Clean up generated files:
```bash
bash cleaner.sh
```

## License

MIT License - Copyright (c) 2026 K0DX-DV1NUX

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.