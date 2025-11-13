"""Text generation utilities for SwitLM"""

import torch
import torch.nn.functional as F


class TextGenerator:
    """Generate text using trained SwitLM models"""
    
    def __init__(self, model, tokenizer):
        """
        Initialize text generator
        
        Args:
            model: Trained SwitLM model
            tokenizer: Tokenizer used during training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(
        self,
        prompt,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        num_return_sequences=1
    ):
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Generated text string (or list of strings if num_return_sequences > 1)
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                current_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # Forward pass
                    outputs = self.model(current_ids)
                    logits = outputs['logits'][:, -1, :]
                    
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for token_id in set(current_ids[0].tolist()):
                            logits[0, token_id] /= repetition_penalty
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # Append to sequence
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decode generated text
                generated_text = self.tokenizer.decode(current_ids[0], skip_special_tokens=True)
                generated_sequences.append(generated_text)
        
        if num_return_sequences == 1:
            return generated_sequences[0]
        return generated_sequences
    
    def generate_batch(self, prompts, **kwargs):
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters (passed to generate())
            
        Returns:
            List of generated text strings
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]