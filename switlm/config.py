"""Configuration classes for SwitLM models"""


class ModelConfig:
    """Configuration for SwitLM model architecture"""
    
    # Predefined model size configurations
    MODEL_SIZES = {
        "50M": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "intermediate_size": 2048,
        },
        "100M": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
        },
        "500M": {
            "hidden_size": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "intermediate_size": 4096,
        },
        "1B": {
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 8192,
        },
        "3B": {
            "hidden_size": 2560,
            "num_layers": 32,
            "num_heads": 32,
            "intermediate_size": 10240,
        },
    }
    
    def __init__(
        self,
        n_parameters="100M",
        num_layers=None,
        hidden_size=None,
        num_heads=None,
        intermediate_size=None,
        vocab_size=50257,  # GPT-2 default
        max_position_embeddings=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
    ):
        """
        Initialize model configuration
        
        Args:
            n_parameters: Model size ("50M", "100M", "500M", "1B", "3B") or custom
            num_layers: Override number of transformer layers
            hidden_size: Override hidden dimension size
            num_heads: Override number of attention heads
            intermediate_size: Override FFN intermediate size
            vocab_size: Vocabulary size
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            layer_norm_eps: Layer normalization epsilon
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        # Get base config from predefined sizes
        if n_parameters in self.MODEL_SIZES:
            base_config = self.MODEL_SIZES[n_parameters]
            self.hidden_size = base_config["hidden_size"]
            self.num_layers = base_config["num_layers"]
            self.num_heads = base_config["num_heads"]
            self.intermediate_size = base_config["intermediate_size"]
            self.model_name = f"SwitLM-{n_parameters}"
        else:
            # Custom configuration
            if not all([num_layers, hidden_size, num_heads, intermediate_size]):
                raise ValueError(
                    f"Unknown model size '{n_parameters}'. "
                    f"Available sizes: {list(self.MODEL_SIZES.keys())} "
                    "or provide custom num_layers, hidden_size, num_heads, intermediate_size"
                )
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.intermediate_size = intermediate_size
            self.model_name = f"SwitLM-Custom-{n_parameters}"
        
        # Override with custom values if provided
        if num_layers is not None:
            self.num_layers = num_layers
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if num_heads is not None:
            self.num_heads = num_heads
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
            
        # Other configurations
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
            
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
    
    def get_model_size(self):
        """Calculate approximate model size in parameters"""
        # Embedding
        embedding_params = self.vocab_size * self.hidden_size
        
        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * (self.hidden_size * self.hidden_size)
        # FFN: gate, up, down
        ffn_params = 3 * (self.hidden_size * self.intermediate_size)
        # Layer norms (2 per layer)
        ln_params = 2 * self.hidden_size
        
        layer_params = attn_params + ffn_params + ln_params
        total_layer_params = layer_params * self.num_layers
        
        # Final layer norm + output projection (tied with embedding)
        final_params = self.hidden_size
        
        total_params = embedding_params + total_layer_params + final_params
        
        return total_params
    
    def __repr__(self):
        return (
            f"ModelConfig(\n"
            f"  model_name={self.model_name},\n"
            f"  parameters={self.get_model_size():,},\n"
            f"  num_layers={self.num_layers},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  intermediate_size={self.intermediate_size}\n"
            f")"
        )


class TrainingConfig:
    """Configuration for training parameters"""
    
    def __init__(
        self,
        learning_rate=3e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        batch_size=None,  # Auto-determined based on model size
        gradient_accumulation_steps=None,  # Auto-determined
        max_length=None,  # Auto-determined
        num_epochs=1,
        save_steps=1000,
        logging_steps=50,
        use_wandb=True,
        wandb_project="switlm-training",
    ):
        """
        Initialize training configuration
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            warmup_ratio: Ratio of warmup steps to total steps
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Batch size (auto-determined if None)
            gradient_accumulation_steps: Gradient accumulation steps (auto-determined if None)
            max_length: Maximum sequence length (auto-determined if None)
            num_epochs: Number of training epochs
            save_steps: Save checkpoint every N steps
            logging_steps: Log metrics every N steps
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
    
    def auto_configure(self, model_config):
        """Auto-configure training parameters based on model size"""
        is_large_model = model_config.hidden_size >= 1024
        
        if self.batch_size is None:
            self.batch_size = 2 if is_large_model else 4
        
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = 16 if is_large_model else 8
        
        if self.max_length is None:
            self.max_length = 384 if is_large_model else 512