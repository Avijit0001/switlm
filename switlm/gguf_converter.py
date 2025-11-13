"""GGUF format conversion for SwitLM models"""

import torch
import struct
import json


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little endian
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3


class GGUFWriter:
    """Writer for GGUF format files"""
    
    def __init__(self, path, arch):
        self.path = path
        self.arch = arch
        self.metadata = {}
        self.tensors = []
        
    def add_metadata(self, key, value):
        """Add metadata key-value pair"""
        self.metadata[key] = value
        
    def add_tensor(self, name, tensor, ggml_type=GGML_TYPE_F32):
        """Add tensor to be written"""
        if ggml_type == GGML_TYPE_F16:
            tensor_data = tensor.to(torch.float16).numpy().tobytes()
        else:
            tensor_data = tensor.to(torch.float32).numpy().tobytes()
            
        self.tensors.append({
            'name': name,
            'shape': list(tensor.shape),
            'type': ggml_type,
            'data': tensor_data
        })
    
    def write_header(self, f):
        """Write GGUF header"""
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(self.tensors)))
        f.write(struct.pack('<Q', len(self.metadata)))
        
    def write_metadata(self, f):
        """Write metadata section"""
        for key, value in self.metadata.items():
            # Key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Value type and data
            if isinstance(value, str):
                f.write(struct.pack('<I', 8))  # String type
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, int):
                f.write(struct.pack('<I', 4))  # Int32 type
                f.write(struct.pack('<i', value))
            elif isinstance(value, float):
                f.write(struct.pack('<I', 5))  # Float32 type
                f.write(struct.pack('<f', value))
            elif isinstance(value, list):
                f.write(struct.pack('<I', 8))  # String type
                value_str = json.dumps(value)
                value_bytes = value_str.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
    
    def write_tensor_info(self, f):
        """Write tensor information section"""
        for tensor in self.tensors:
            # Name
            name_bytes = tensor['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Dimensions
            f.write(struct.pack('<I', len(tensor['shape'])))
            for dim in tensor['shape']:
                f.write(struct.pack('<Q', dim))
                
            # Type
            f.write(struct.pack('<I', tensor['type']))
            
            # Offset (placeholder)
            f.write(struct.pack('<Q', 0))
            
    def write(self):
        """Write complete GGUF file"""
        with open(self.path, 'wb') as f:
            self.write_header(f)
            self.write_metadata(f)
            self.write_tensor_info(f)
            
            # Align to 32 bytes
            current_pos = f.tell()
            padding = (32 - (current_pos % 32)) % 32
            f.write(b'\x00' * padding)
            
            # Write tensor data
            for tensor in self.tensors:
                f.write(tensor['data'])


def convert_to_gguf(model_path, output_path):
    """
    Convert PyTorch model to GGUF format
    
    Args:
        model_path: Path to PyTorch checkpoint (.pt file)
        output_path: Path for output GGUF file
    """
    print(f"🔄 Converting model to GGUF format...")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        # Add safe globals for older PyTorch versions
        import torch.serialization
        from .config import ModelConfig
        torch.serialization.add_safe_globals([ModelConfig])
        checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['config']
    model_state = checkpoint['model_state_dict']
    
    # Determine architecture name
    arch_name = "switlm"
    
    # Initialize GGUF writer
    writer = GGUFWriter(output_path, arch_name)
    
    # Add metadata
    writer.add_metadata("general.architecture", arch_name)
    writer.add_metadata("general.name", config.model_name)
    writer.add_metadata(f"{arch_name}.context_length", config.max_position_embeddings)
    writer.add_metadata(f"{arch_name}.embedding_length", config.hidden_size)
    writer.add_metadata(f"{arch_name}.block_count", config.num_layers)
    writer.add_metadata(f"{arch_name}.attention.head_count", config.num_heads)
    writer.add_metadata(f"{arch_name}.attention.head_count_kv", config.num_heads)
    writer.add_metadata(f"{arch_name}.feed_forward_length", config.intermediate_size)
    writer.add_metadata("general.file_type", 1)  # F16
    writer.add_metadata("tokenizer.ggml.model", "gpt2")
    
    print(f"📦 Converting {len(model_state)} tensors...")
    
    # Convert model tensors to GGUF format
    for name, tensor in model_state.items():
        gguf_name = _convert_tensor_name(name)
        writer.add_tensor(gguf_name, tensor, GGML_TYPE_F16)
    
    # Write GGUF file
    writer.write()
    print(f"✅ GGUF conversion complete!")


def _convert_tensor_name(pytorch_name):
    """Convert PyTorch tensor name to GGUF format"""
    if 'embed_tokens.weight' in pytorch_name:
        return 'token_embd.weight'
    elif 'lm_head.weight' in pytorch_name:
        return 'output.weight'
    elif 'norm.weight' in pytorch_name:
        return 'output_norm.weight'
    elif 'layers.' in pytorch_name:
        parts = pytorch_name.split('.')
        layer_num = parts[1]
        
        if 'ln_1.weight' in pytorch_name:
            return f'blk.{layer_num}.attn_norm.weight'
        elif 'ln_2.weight' in pytorch_name:
            return f'blk.{layer_num}.ffn_norm.weight'
        elif 'attn.q_proj.weight' in pytorch_name:
            return f'blk.{layer_num}.attn_q.weight'
        elif 'attn.k_proj.weight' in pytorch_name:
            return f'blk.{layer_num}.attn_k.weight'
        elif 'attn.v_proj.weight' in pytorch_name:
            return f'blk.{layer_num}.attn_v.weight'
        elif 'attn.out_proj.weight' in pytorch_name:
            return f'blk.{layer_num}.attn_output.weight'
        elif 'mlp.gate.weight' in pytorch_name:
            return f'blk.{layer_num}.ffn_gate.weight'
        elif 'mlp.up.weight' in pytorch_name:
            return f'blk.{layer_num}.ffn_up.weight'
        elif 'mlp.down.weight' in pytorch_name:
            return f'blk.{layer_num}.ffn_down.weight'
    
    return pytorch_name