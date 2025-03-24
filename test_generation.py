import torch
import time
import os
import shutil
import argparse
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_sparsity(model, sparsity):
    for module in model.modules():
        if hasattr(module, "__class__") and "AttentionExperimental" in module.__class__.__name__:
            module.token_sparse_method = sparsity
            module.set_token_sparsity()
    return model

def prepare_model(hf_model_name, local_model_dir="./downloaded_models", force_download=False):
    """
    Downloads the model from Hugging Face and replaces modeling_llama_butler.py
    with the local version. If the model already exists locally, it just updates
    the modeling_llama_butler.py file.
    
    Args:
        hf_model_name: Name of the model on Hugging Face Hub or local path
        local_model_dir: Directory to save the model locally
        force_download: Whether to force re-download even if model exists locally
    
    Returns:
        Path to the prepared model
    """
    print(f"Preparing model: {hf_model_name}")
    
    # Create the local model directory if it doesn't exist
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Determine the model cache path
    model_cache_path = os.path.join(local_model_dir, os.path.basename(hf_model_name))
    
    # Check if model already exists locally
    if not os.path.exists(model_cache_path) or force_download:
        print(f"Downloading model from Hugging Face: {hf_model_name}")
        # Download model files from Hugging Face
        snapshot_download(
            repo_id=hf_model_name,
            local_dir=model_cache_path,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to: {model_cache_path}")
    else:
        print(f"Using existing model from: {model_cache_path}")
    
    # Path to the local modeling_llama_butler.py
    local_butler_path = "./modeling_llama_butler.py"
    
    # Path where it should be in the model directory
    target_butler_path = os.path.join(model_cache_path, "modeling_llama_butler.py")
    
    # Copy the local modeling_llama_butler.py to the model directory
    if os.path.exists(local_butler_path):
        print(f"Replacing modeling_llama_butler.py with local version")
        shutil.copy2(local_butler_path, target_butler_path)
    else:
        print(f"Warning: Local modeling_llama_butler.py not found at {local_butler_path}")
    
    return model_cache_path

def test_generation(model_path=None, baseline_model_path=None, prompt=None, num_tokens=100, temperature=0.7, top_p=0.9, sparsity=None, seed=42):
    """
    Test text generation with the specified model.
    
    Args:
        model_path: Path to the model or Hugging Face model name
        prompt: Text prompt for generation
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    if model_path is None:
        model_path = "./"  # Default to local directory
    
    # If model_path looks like a Hugging Face model name, prepare it locally
    if "/" in model_path and not os.path.exists(model_path):
        model_path = prepare_model(model_path)
    
    # Default prompt if not provided
    if prompt is None:
        prompt = "If millionaires have butlers, why don't million dollar language models have a butler too? I think its because "
    
    num_tokens_to_generate = num_tokens
    
    print("Loading model and tokenizer...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if sparsity is not None:
        model = set_sparsity(model, sparsity)
    else:
        model = set_sparsity(model, "fixed_50pc")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = len(inputs.input_ids[0])
    
    # Warm up the model
    print("Warming up the model...")
    with torch.no_grad():
        _ = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False
        )
    
    # Measure generation speed
    print(f"Generating {num_tokens_to_generate} tokens...")
    # Start timing
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=num_tokens_to_generate,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None
        )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate number of tokens generated
    generated_ids = output.sequences[0][input_length:]
    num_generated_tokens = len(generated_ids)
    
    # Decode generated text
    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    
    if baseline_model_path is not None:
        # Load baseline model
        print("Loading baseline model...")
        baseline_model = AutoModelForCausalLM.from_pretrained(baseline_model_path, trust_remote_code=True)
        baseline_model = baseline_model.to(device)
        
        # Warm up the baseline model
        print("Warming up the baseline model...")
        baseline_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = baseline_model.generate(
                baseline_inputs.input_ids,
                max_new_tokens=10,
                do_sample=False
            )
        
        # Measure baseline generation speed
        print(f"Generating {num_tokens_to_generate} tokens with baseline model...")
        baseline_start_time = time.time()
        
        # Generate text with baseline model
        with torch.no_grad():
            baseline_output = baseline_model.generate(
                baseline_inputs.input_ids,
                max_new_tokens=num_tokens_to_generate,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=None
            )
        
        # Calculate baseline total time
        baseline_total_time = time.time() - baseline_start_time
        
        # Calculate number of tokens generated by baseline
        baseline_input_length = len(baseline_inputs.input_ids[0])
        baseline_generated_ids = baseline_output.sequences[0][baseline_input_length:]
        baseline_num_generated_tokens = len(baseline_generated_ids)
        
        # Decode baseline generated text
        baseline_generated_text = tokenizer.decode(baseline_output.sequences[0], skip_special_tokens=True)
        
        # Calculate tokens per second (throughput) for both models
        tokens_per_second = num_generated_tokens / total_time
        baseline_tokens_per_second = baseline_num_generated_tokens / baseline_total_time
        
        # Calculate speedup based on tokens per second (throughput)
        # This is the preferred metric as it accounts for any differences in the actual number of tokens generated
        throughput_speedup = tokens_per_second / baseline_tokens_per_second

    # Print results
    print("\n--- TokenButler Generated Text ---")
    print(generated_text)
    if baseline_model_path is not None:
        print("\n--- Baseline Generated Text ---")
        print(baseline_generated_text)
    print("\n--- TokenButler Performance ---")
    print(f"Prompt length: {input_length}")
    print(f"Number of tokens generated: {num_generated_tokens}")
    print(f"Total generation time: {total_time:.3f} seconds")
    print(f"Tokens per second (tokens/s): {num_generated_tokens / total_time:.2f}")
    print(f"Average time per token: {(total_time / num_generated_tokens) * 1000:.2f} ms")
    
    if baseline_model_path is not None:
        print("\n--- Baseline Model Performance ---")
        print(f'Number of tokens generated: {baseline_num_generated_tokens}')
        print(f"Baseline total generation time: {baseline_total_time:.3f} seconds")
        print(f"Baseline tokens per second: {baseline_tokens_per_second:.2f}")
        print(f"Speedup (tokens/s): {throughput_speedup:.2f}x")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test text generation with a Hugging Face model")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to local model or Hugging Face model name")
    parser.add_argument("--baseline_model_path", type=str, default=None,
                        help="Path to local baseline model or Hugging Face model name")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--num_tokens", type=int, default=256,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--force_download", action="store_true",
                        help="Force re-download of model even if it exists locally")
    parser.add_argument("--sparsity", type=str, default="fixed_50pc",
                        help="Sparsity level for TokenButler")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # If it's a Hugging Face model, prepare it first
    if args.model_path and "/" in args.model_path and not os.path.exists(args.model_path):
        args.model_path = prepare_model(args.model_path, force_download=args.force_download)
    
    # Run the generation test
    test_generation(
        model_path=args.model_path,
        baseline_model_path=args.baseline_model_path,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sparsity=args.sparsity,
        seed=args.seed
    )