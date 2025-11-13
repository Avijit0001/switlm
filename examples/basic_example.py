"""
Basic usage example for SwitLM
"""

from switlm import SwitLMTrainer

def basic_training():
    """Train a simple model"""
    print("="*60)
    print("Basic Training Example")
    print("="*60)
    
    # Create trainer with 100M parameter model
    trainer = SwitLMTrainer(
        n_parameters="100M",
        dataset="wikitext",
        num_layers=12
    )
    
    # Train the model
    trainer.train()
    
    # Save as GGUF
    trainer.save("basic_model.gguf")
    
    # Generate some text
    print("\n" + "="*60)
    print("Generating Text")
    print("="*60)
    
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important invention in history"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = trainer.generate(prompt, max_length=50, temperature=0.7)
        print(f"Generated: {generated}")


def multi_dataset_training():
    """Train on multiple datasets"""
    print("\n" + "="*60)
    print("Multi-Dataset Training Example")
    print("="*60)
    
    # Train on multiple datasets
    trainer = SwitLMTrainer(
        n_parameters="500M",
        dataset=["wikitext", "ag_news", "imdb"],
        num_layers=16
    )
    
    trainer.train()
    trainer.save("multi_dataset_model.gguf")


def custom_configuration():
    """Custom model and training configuration"""
    print("\n" + "="*60)
    print("Custom Configuration Example")
    print("="*60)
    
    from switlm import ModelConfig, TrainingConfig
    
    # Custom model architecture
    model_config = ModelConfig(
        n_parameters="custom",
        num_layers=18,
        hidden_size=1280,
        num_heads=10,
        intermediate_size=5120
    )
    
    # Custom training settings
    training_config = TrainingConfig(
        learning_rate=2e-4,
        batch_size=2,
        gradient_accumulation_steps=16,
        num_epochs=2,
        use_wandb=False  # Disable W&B
    )
    
    trainer = SwitLMTrainer(
        model_config=model_config,
        training_config=training_config,
        dataset="wikitext"
    )
    
    trainer.train()
    trainer.save("custom_model.gguf")


def load_and_continue():
    """Load existing model and continue training"""
    print("\n" + "="*60)
    print("Load and Continue Training Example")
    print("="*60)
    
    # Create initial model
    trainer = SwitLMTrainer(
        n_parameters="100M",
        dataset="wikitext"
    )
    trainer.train()
    trainer.save("checkpoint.pt", format="pytorch")
    
    # Load and continue training
    new_trainer = SwitLMTrainer(n_parameters="100M")
    new_trainer.load("checkpoint.pt")
    
    # Train on new dataset
    new_trainer.datasets = ["ag_news"]
    new_trainer.train(num_epochs=1)
    new_trainer.save("continued_model.gguf")


if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic Training", basic_training),
        "2": ("Multi-Dataset Training", multi_dataset_training),
        "3": ("Custom Configuration", custom_configuration),
        "4": ("Load and Continue", load_and_continue),
    }
    
    print("SwitLM Examples")
    print("="*60)
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect example (1-4) or 'all': ").strip()
    
    if choice == "all":
        for name, func in examples.values():
            func()
    elif choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("Running basic example...")
        basic_training()