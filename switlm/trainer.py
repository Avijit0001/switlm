"""Main training interface for SwitLM"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import gc
import warnings

from .config import ModelConfig, TrainingConfig
from .model import SwitLMModel
from .dataset import load_dataset_texts, TextDataset
from .gguf_converter import convert_to_gguf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not installed. Install with 'pip install wandb' for experiment tracking.")


class SwitLMTrainer:
    """
    Main trainer class for SwitLM models
    
    Example:
        >>> from switlm import SwitLMTrainer
        >>> trainer = SwitLMTrainer(n_parameters="1B", dataset="wikitext", num_layers=10)
        >>> trainer.train()
        >>> trainer.save("my_model.gguf")
    """
    
    def __init__(
        self,
        n_parameters="100M",
        dataset="wikitext",
        num_layers=None,
        hidden_size=None,
        num_heads=None,
        intermediate_size=None,
        tokenizer_name="gpt2",
        device=None,
        model_config=None,
        training_config=None,
        **kwargs
    ):
        """
        Initialize SwitLM trainer
        
        Args:
            n_parameters: Model size ("50M", "100M", "500M", "1B", "3B")
            dataset: Dataset name or list of dataset names
                Available: "wikitext", "ag_news", "imdb", "squad", "tiny_stories", etc.
            num_layers: Override number of transformer layers
            hidden_size: Override hidden dimension size
            num_heads: Override number of attention heads
            intermediate_size: Override FFN intermediate size
            tokenizer_name: HuggingFace tokenizer name (default: "gpt2")
            device: Training device (auto-detected if None)
            model_config: Custom ModelConfig object
            training_config: Custom TrainingConfig object
            **kwargs: Additional training configuration parameters
        """
        # Setup device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Initialize configurations
        if model_config is None:
            self.model_config = ModelConfig(
                n_parameters=n_parameters,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size
            )
        else:
            self.model_config = model_config
        
        if training_config is None:
            self.training_config = TrainingConfig(**kwargs)
        else:
            self.training_config = training_config
        
        # Auto-configure training parameters
        self.training_config.auto_configure(self.model_config)
        
        # Load tokenizer
        print(f"📚 Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update model config with tokenizer info
        self.model_config.vocab_size = len(self.tokenizer)
        self.model_config.pad_token_id = self.tokenizer.pad_token_id
        
        # Print model info
        print(f"\n{self.model_config}")
        print(f"💾 Estimated parameters: {self.model_config.get_model_size():,}")
        
        # Initialize model
        print(f"\n🏗️  Building model...")
        self.model = SwitLMModel(self.model_config).to(self.device)
        
        # Count actual parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
        
        # Setup dataset
        if isinstance(dataset, str):
            self.datasets = [dataset]
        else:
            self.datasets = dataset
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.trained = False
        
        # Setup W&B if available
        if self.training_config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.training_config.wandb_project,
                config={
                    "model_size": n_parameters,
                    "datasets": self.datasets,
                    "model_config": vars(self.model_config),
                    "training_config": vars(self.training_config),
                }
            )
    
    def _setup_optimizer_scheduler(self, total_steps):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(self.training_config.beta1, self.training_config.beta2)
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.training_config.warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
    
    def _train_on_dataset(self, dataset_name, epoch_global):
        """Train model on a single dataset"""
        print(f"\n{'='*60}")
        print(f"📖 Loading dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset_texts = load_dataset_texts(dataset_name)
        
        if not dataset_texts or len(dataset_texts) == 0:
            print(f"⚠️  Skipping {dataset_name} - no data loaded")
            return 0
        
        print(f"✅ Loaded {len(dataset_texts)} samples from {dataset_name}")
        
        # Create dataset and dataloader
        dataset = TextDataset(
            dataset_texts, 
            self.tokenizer, 
            max_length=self.training_config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        accumulation_steps = self.training_config.gradient_accumulation_steps
        
        progress_bar = tqdm(dataloader, desc=f"Training on {dataset_name}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cuda'):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
            
            # Logging
            if step % self.training_config.logging_steps == 0:
                if self.training_config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'loss': loss.item() * accumulation_steps,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'dataset': dataset_name,
                        'step': step + epoch_global * 1000
                    })
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Completed {dataset_name}. Average loss: {avg_loss:.4f}")
        
        # Cleanup
        del dataset, dataloader, dataset_texts
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_loss
    
    def train(self, num_epochs=None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train (uses config default if None)
        """
        if num_epochs is None:
            num_epochs = self.training_config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting training on {len(self.datasets)} dataset(s)")
        print(f"{'='*60}\n")
        
        # Estimate total steps
        estimated_steps_per_dataset = 2000
        total_steps = len(self.datasets) * estimated_steps_per_dataset * num_epochs
        
        # Setup optimizer and scheduler
        self._setup_optimizer_scheduler(total_steps)
        
        # Train on each dataset
        epoch_global = 0
        for epoch in range(num_epochs):
            for dataset_name in self.datasets:
                try:
                    avg_loss = self._train_on_dataset(dataset_name, epoch_global)
                    
                    # Log dataset completion
                    if self.training_config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'dataset_completed': dataset_name,
                            'dataset_loss': avg_loss,
                            'epoch': epoch
                        })
                    
                    epoch_global += 1
                    
                except Exception as e:
                    print(f"❌ Error training on {dataset_name}: {str(e)}")
                    print("Continuing with next dataset...")
                    torch.cuda.empty_cache()
                    gc.collect()
        
        self.trained = True
        print(f"\n{'='*60}")
        print(f"🎉 Training completed successfully!")
        print(f"{'='*60}\n")
    
    def save(self, output_path, format="gguf"):
        """
        Save the trained model
        
        Args:
            output_path: Path to save the model
            format: Output format ("gguf", "pytorch", or "both")
        """
        if not self.trained:
            warnings.warn("Model has not been trained yet. Saving untrained model.")
        
        print(f"\n💾 Saving model to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save PyTorch checkpoint
        if format in ["pytorch", "both"]:
            pt_path = output_path if output_path.endswith('.pt') else output_path.replace('.gguf', '.pt')
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config,
                'tokenizer_name': 'gpt2'
            }
            torch.save(checkpoint, pt_path)
            print(f"✅ PyTorch model saved to: {pt_path}")
        
        # Save GGUF format
        if format in ["gguf", "both"]:
            gguf_path = output_path if output_path.endswith('.gguf') else output_path + '.gguf'
            
            # Save temporary PyTorch checkpoint for conversion
            temp_pt_path = "temp_checkpoint.pt"
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model_config,
                'tokenizer_name': 'gpt2'
            }
            torch.save(checkpoint, temp_pt_path)
            
            # Convert to GGUF
            convert_to_gguf(temp_pt_path, gguf_path)
            
            # Cleanup temp file
            if os.path.exists(temp_pt_path):
                os.remove(temp_pt_path)
            
            print(f"✅ GGUF model saved to: {gguf_path}")
        
        print(f"✅ Model saved successfully!")
    
    def load(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved PyTorch model (.pt file)
        """
        print(f"📥 Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        self.model_config = checkpoint['config']
        
        # Rebuild model with loaded config
        self.model = SwitLMModel(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.trained = True
        print(f"✅ Model loaded successfully!")
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        """
        Generate text using the trained model
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text string
        """
        from .generator import TextGenerator
        
        generator = TextGenerator(self.model, self.tokenizer)
        return generator.generate(
            prompt, 
            max_length=max_length, 
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    
    def __repr__(self):
        return (
            f"SwitLMTrainer(\n"
            f"  model={self.model_config.model_name},\n"
            f"  parameters={self.model_config.get_model_size():,},\n"
            f"  datasets={self.datasets},\n"
            f"  device={self.device},\n"
            f"  trained={self.trained}\n"
            f")"
        )