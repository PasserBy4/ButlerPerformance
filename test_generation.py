import torch
import time
import os
import shutil
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

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

def test_generation(model_path=None, prompt=None, num_tokens=100, temperature=0.7, top_p=0.9):
    """
    Test text generation with the specified model.
    
    Args:
        model_path: Path to the model or Hugging Face model name
        prompt: Text prompt for generation
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    """
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
            output_scores=True
        )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate number of tokens generated
    generated_ids = output.sequences[0][input_length:]
    num_generated_tokens = len(generated_ids)
    
    # Decode generated text
    generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    
    # Print results
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n--- Performance Metrics ---")
    print(f"Total generation time: {total_time:.3f} seconds")
    print(f"Number of tokens generated: {num_generated_tokens}")
    print(f"Tokens per second (tokens/s): {num_generated_tokens / total_time:.2f}")
    print(f"Average time per token: {(total_time / num_generated_tokens) * 1000:.2f} ms")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test text generation with a Hugging Face model")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to local model or Hugging Face model name")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--num_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--force_download", action="store_true",
                        help="Force re-download of model even if it exists locally")
    
    args = parser.parse_args()
    
    # If it's a Hugging Face model, prepare it first
    if args.model_path and "/" in args.model_path and not os.path.exists(args.model_path):
        args.model_path = prepare_model(args.model_path, force_download=args.force_download)
    
    # Run the generation test
    test_generation(
        model_path=args.model_path,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )