# Neural Sparse Attention (NSA) Model

This repository contains an implementation of a Neural Sparse Attention (NSA) model, an efficient transformer-based architecture that uses various attention mechanisms to optimize memory usage and computational efficiency while maintaining performance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Architecture](#architecture)
- [Usage](#usage)
- [Training](#training)
- [Monitoring and Visualization](#monitoring-and-visualization)
- [Tokenizer](#tokenizer)
- [License](#license)

## Overview

The NSA model implements a transformer architecture with specialized attention mechanisms that reduce memory and computational requirements by using sparse attention patterns. This makes it suitable for training on longer sequences with limited computational resources.

This implementation is based on the Neural Sparse Attention model described in the paper:  
[Neural Sparse Attention](https://arxiv.org/abs/2502.11089)

The implementation includes:
- A core model with customizable neural sparse attention
- A two-phase training pipeline
- Memory-efficient training with gradient accumulation and mixed precision
- Dynamic sparsity control that adjusts attention patterns based on input complexity
- Mixture of Experts (MoE) implementation for more efficient parameter usage
- Comprehensive training monitoring and visualization tools

## Features

- **Multiple Attention Mechanisms**:
  - Compression Attention: Reduces sequence length by merging tokens
  - Selection Attention: Focuses on the most important tokens
  - Window Attention: Local attention within sliding windows

- **Dynamic Sparsity Control**: Automatically adjusts sparsity parameters based on input complexity

- **Mixture of Experts (MoE)**: Routes computation through specialized expert networks

- **Optimized Training**:
  - Automatic batch size adjustment based on available GPU memory
  - Gradient accumulation for larger effective batch sizes
  - Mixed precision training
  - Warm-up learning rate scheduling
  - Memory monitoring and OOM recovery
  - Early stopping

- **Comprehensive Monitoring**:
  - Real-time training metrics logging
  - GPU memory usage tracking
  - Dynamic sparsity parameter visualization
  - Training/validation loss comparison
  - Two-phase training comparison

## Requirements

The project requires several Python packages to run properly. You can install all dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch for deep learning
- Transformers for model architecture
- NumPy and Pandas for data processing
- Matplotlib for visualization
- SentencePiece for tokenization
- Tkinter for GUI monitoring tools

For a complete list with version specifications, see the [requirements.txt](requirements.txt) file.


## Architecture

### Core Components

1. **NSAModel**: The main model class that integrates all components
2. **NSAAttentionExtended**: Implementation of neural sparse attention mechanisms
3. **NSABlockExtended**: Transformer block with sparse attention and feed-forward layers
4. **OptimizedTrainer**: Memory-efficient training implementation
5. **DynamicSparsityController**: Controls sparsity parameters adaptively
6. **NSAAttentionExtendedWithRouting**: MoE implementation for routing through experts

### Key Parameters

- `hidden_size`: Size of hidden layers
- `num_attention_heads`: Number of attention heads
- `num_hidden_layers`: Number of transformer layers
- `compress_ratio`: Compression rate for compressed attention
- `select_k`: Number of tokens to select in selection attention
- `window_size`: Size of local attention window
- `use_dynamic_sparsity`: Enable/disable dynamic parameter adjustment

## Usage

### Setting up the Model

```python
# Initialize configuration
config = NSAConfig(
    vocab_size=16000,
    max_seq_length=512,
    hidden_size=896,
    num_attention_heads=8,
    num_hidden_layers=8,
    compress_ratio=4,
    select_k=16,
    window_size=64,
    use_dynamic_sparsity=True
)

# Create model
model = NSAModel(config)
```

### Tokenization

```python
# Initialize tokenizer
tokenizer = EnhancedSPTokenizer(vocab_size=16000)

# Train tokenizer on your data
tokenizer.train(text_samples, model_prefix="my_tokenizer")

# Tokenize text
encoding = tokenizer("Your text here", max_length=512, return_tensors="pt")
```

### Training

```python
# Prepare dataset
train_dataset = ChineseTextDataset(train_data, tokenizer)
test_dataset = ChineseTextDataset(test_data, tokenizer)

# Create trainer
trainer = OptimizedTrainerWithLogging(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    learning_rate=5e-5,
    num_epochs=3,
    mixed_precision=True,
    model_name="my_model",
    log_dir="training_logs"
)

# Train model
trainer.train()

# Save model
trainer.save_model("model_checkpoint.pth")
```

## Training

The implementation supports a two-phase training approach:

1. **Phase 1**: Fixed sparsity parameters
   - Higher learning rate
   - Longer warmup period
   - Fixed attention sparsity

2. **Phase 2**: Dynamic sparsity
   - Lower learning rate
   - Shorter warmup period
   - Dynamically adjusted sparsity based on input complexity

This approach allows the model to first learn basic patterns with fixed parameters, then fine-tune with adaptive parameters.

## Monitoring and Visualization

### Training Visualization

The code includes comprehensive visualization tools through the `TrainingVisualizer` class:

```python
# Initialize visualizer
visualizer = TrainingVisualizer(log_dir="training_logs")

# Find log files
log_files = visualizer.find_log_files()

# Load a specific log
visualizer.load_log(log_files[0])

# Generate plots
visualizer.plot_all(output_dir="plots", show=True)

# Print statistics
visualizer.print_stats()
```

### Real-time Sparsity Monitoring

The `SparsityMonitor` class provides a GUI to monitor dynamic sparsity parameters in real-time:

```python
# Initialize and run monitor
monitor = SparsityMonitor(model_name="NSA Model", log_file="sparsity_metrics.json")
monitor.run()
```

## Tokenizer

The implementation includes an `EnhancedSPTokenizer` class based on SentencePiece for Chinese text tokenization:

- Supports BPE or Unigram models
- Special token handling
- Configurable vocabulary size
- Integrated with PyTorch for seamless dataset creation

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
