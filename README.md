# SwitLM - Simple Witty Language Model Training

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SwitLM** is a simple yet powerful Python library for creating and training custom Language Models with minimal code. Train your own LLM and export to GGUF format in just a few lines!

## ✨ Features

- 🚀 **Simple API**: Create and train LLMs with 3 lines of code
- 🎯 **Multiple Model Sizes**: From 50M to 3B parameters
- 🏗️ **Modern Architecture**: RoPE, RMSNorm, SwiGLU (LLaMA-style)
- 📊 **Rich Dataset Support**: WikiText, AG News, IMDB, SQuAD, TinyStories, and more
- 💾 **GGUF Export**: Direct export to GGUF format for llama.cpp
- ⚡ **GPU Optimized**: Memory-efficient training with gradient checkpointing
- 📈 **W&B Integration**: Built-in experiment tracking
- 🎨 **Flexible Configuration**: Easy customization of architecture and training

## 📦 Installation

```bash
pip install switlm
```

Or install from source:

```bash
git clone https://github.com/Avijit0001/switlm.git
cd switlm
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0

## 🚀 Quick Start

### Basic Usage

```python
from switlm import SwitLMTrainer

# Create and train a 1B parameter model
trainer = SwitLMTrainer(
    n_parameters="1B",
    dataset="wikitext",
    num_layers=24
)

# Train the model
trainer.train()

# Save as GGUF (ready for llama.cpp)
trainer.save("my_model.gguf")
```

### Generate Text

```python
# Generate text with your trained model
output = trainer.generate(
    "The future of artificial intelligence",
    max_length=100,
    temperature=0.7
)
print(output)
```

### Multiple Datasets

```python
# Train on multiple datasets sequentially
trainer = SwitLMTrainer(
    n_parameters="500M",
    dataset=["wikitext", "ag_news", "imdb"],
    num_layers=16
)
trainer.train()
trainer.save("multi_dataset_model.gguf")
```

### Custom Configuration

```python
from switlm import SwitLMTrainer, ModelConfig, TrainingConfig

# Custom model architecture
model_config = ModelConfig(
    n_parameters="custom",
    num_layers=20,
    hidden_size=1536,
    num_heads=12,
    intermediate_size=6144
)

# Custom training settings
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=3,
    use_wandb=True
)

trainer = SwitLMTrainer(
    model_config=model_config,
    training_config=training_config,
    dataset="wikitext"
)

trainer.train()
trainer.save("custom_model.gguf")
```

## 🎯 Available Model Sizes

| Size | Parameters | Layers | Hidden Size | Heads | Use Case |
|------|-----------|--------|-------------|-------|----------|
| 50M | ~50M | 8 | 512 | 8 | Quick experiments |
| 100M | ~100M | 12 | 768 | 12 | Small projects |
| 500M | ~500M | 16 | 1024 | 16 | Medium tasks |
| 1B | ~1B | 24 | 2048 | 16 | Serious applications |
| 3B | ~3B | 32 | 2560 | 32 | Production use |

## 📚 Supported Datasets

- `wikitext` - Wikipedia articles
- `ag_news` - News classification
- `imdb` - Movie reviews
- `squad` - Question answering
- `tiny_stories` - Short stories
- `openwebtext` - Web text
- `c4` - Colossal Clean Crawled Corpus
- `bookcorpus` - Books
- `pile` - Diverse text corpus

## 🔧 Advanced Features

### Load and Continue Training

```python
# Load a previously trained model
trainer = SwitLMTrainer(n_parameters="1B")
trainer.load("my_model.pt")

# Continue training on new data
trainer.datasets = ["squad", "imdb"]
trainer.train(num_epochs=2)
trainer.save("continued_model.gguf")
```

### Custom Text Generation

```python
from switlm import TextGenerator

# Create generator
generator = TextGenerator(trainer.model, trainer.tokenizer)

# Generate with custom parameters
text = generator.generate(
    "Once upon a time",
    max_length=200,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2
)
print(text)
```

### Save in Multiple Formats

```python
# Save as both PyTorch and GGUF
trainer.save("my_model", format="both")

# Outputs:
# - my_model.pt (PyTorch checkpoint)
# - my_model.gguf (GGUF format)
```

## 🏗️ Architecture Details

SwitLM implements a modern transformer architecture with:

- **RoPE (Rotary Position Embeddings)**: Better positional encoding
- **RMSNorm**: More stable training than LayerNorm
- **SwiGLU**: Advanced activation function (from PaLM/LLaMA)
- **Pre-norm Architecture**: Better gradient flow
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision**: Faster training with FP16

## 📊 Training Configuration

```python
TrainingConfig(
    learning_rate=3e-4,         # Learning rate
    weight_decay=0.01,          # Weight decay for regularization
    beta1=0.9,                  # Adam beta1
    beta2=0.95,                 # Adam beta2
    warmup_ratio=0.1,           # Warmup ratio
    max_grad_norm=1.0,          # Gradient clipping
    batch_size=4,               # Batch size (auto if None)
    gradient_accumulation_steps=8,  # Accumulation steps
    max_length=512,             # Max sequence length
    num_epochs=1,               # Number of epochs
    use_wandb=True,             # W&B logging
    wandb_project="switlm"      # W&B project name
)
```

## 💡 Tips and Best Practices

1. **Start Small**: Begin with 50M or 100M models to test your pipeline
2. **GPU Memory**: Larger models require more VRAM (1B model needs ~12GB)
3. **Dataset Size**: More data generally means better models
4. **Learning Rate**: Start with 3e-4, adjust based on loss curves
5. **Sequence Length**: Shorter sequences (256-512) train faster
6. **Gradient Accumulation**: Increase if you run out of memory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by modern LLM architectures (LLaMA, GPT, PaLM)
- Built with PyTorch and HuggingFace Transformers
- GGUF format support for llama.cpp integration


## 🌟 Star History

If you find SwitLM useful, please consider giving it a star! ⭐

---

Made with ❤️ by the SwitLM team (Avijit Paul)