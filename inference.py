import argparse
import json
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel

# Set default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported models
SUPPORTED_MODELS = {
    "gemma2": {
        "base": "google/gemma-2b",
        "chat_template": True,
        "is_chat_model": False  # Base model is not a chat model
    },
    "qwen2.5": {
        "base": "Qwen/Qwen2.5-0.5B",
        "chat_template": True,
        "is_chat_model": True  # Qwen2.5 is designed as a chat model
    }
}


def load_tokenizer(model_path: str, trust_remote_code: bool = False) -> PreTrainedTokenizer:
    """Load tokenizer from the given model path.

    Args:
        model_path: Path to the model or model identifier from huggingface.co/models
        trust_remote_code: Whether to trust remote code when loading the tokenizer

    Returns:
        The loaded tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        # Set padding token if not set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"

        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise


def load_base_model(model_path: str, model_type: str = "gemma2",
                  dtype: str = "auto", trust_remote_code: bool = True) -> PreTrainedModel:
    """Load base model from the given model path.

    Args:
        model_path: Path to the model or model identifier from huggingface.co/models
        model_type: Type of the model (gemma2, qwen2.5)
        dtype: Data type for model weights (auto, float16, bfloat16, float32)
        trust_remote_code: Whether to trust remote code when loading the model

    Returns:
        The loaded model
    """
    # Determine compute dtype
    if dtype == "auto":
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif dtype == "float16":
        compute_dtype = torch.float16
    elif dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:  # float32
        compute_dtype = torch.float32

    # Load model
    try:
        # For Gemma 2 and Qwen 2.5, we always use AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            trust_remote_code=trust_remote_code,  # Both models require trust_remote_code=True
            device_map="auto" if DEVICE == "cuda" else None
        )

        # Print model info
        print(f"Loaded {model.__class__.__name__} model")
        print(f"Model parameters: {model.num_parameters():,}")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_adapter(model: PreTrainedModel, adapter_path: str, adapter_type: str = "lora") -> PreTrainedModel:
    """Load and apply adapter to the base model.

    Args:
        model: Base model to apply adapter to
        adapter_path: Path to the adapter or adapter identifier from huggingface.co/models
        adapter_type: Type of the adapter (lora, prefix, prompt_tuning, etc.)

    Returns:
        Model with adapter applied
    """
    try:
        # Currently only supporting PEFT adapters
        if adapter_type in ["lora", "prefix", "prompt_tuning", "ia3"]:
            # Load PEFT model
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"Loaded {adapter_type} adapter from {adapter_path}")
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        return model
    except Exception as e:
        print(f"Error loading adapter: {e}")
        raise


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> str:
    """Generate text using the model.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for generation
        prompt: Input prompt for generation
        generation_config: Configuration for generation

    Returns:
        Generated text
    """
    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )

    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If the model is an encoder-decoder, the generated text won't include the prompt
    # For causal LMs, we need to remove the prompt from the output
    if isinstance(model, AutoModelForCausalLM):
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

    return generated_text


def chat_completion(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                   messages: List[Dict[str, str]], model_type: str,
                   generation_config: Optional[Dict[str, Any]] = None,
                   use_chat: bool = False) -> str:
    """Generate chat completion using the model.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for generation
        messages: List of message dictionaries with 'role' and 'content' keys
        model_type: Type of the model (gemma2, qwen2.5)
        generation_config: Configuration for generation
        use_chat: Whether to use chat format (for chat models) or not (for base models)

    Returns:
        Generated assistant response
    """
    # Check if the model has a chat template
    has_chat_template = hasattr(tokenizer, "apply_chat_template") and SUPPORTED_MODELS.get(model_type, {}).get("chat_template", False)
    is_chat_model = use_chat or SUPPORTED_MODELS.get(model_type, {}).get("is_chat_model", False)

    if has_chat_template and is_chat_model:
        # Use the model's built-in chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to a simple template if the model doesn't have a chat template or is not a chat model
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Add the assistant prefix for the response
        prompt += "Assistant: "

    # Generate response
    response = generate_text(model, tokenizer, prompt, generation_config)

    return response


def batch_generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                  prompts: List[str], generation_config: Optional[Dict[str, Any]] = None,
                  batch_size: int = 1) -> List[str]:
    """Generate text for multiple prompts in batches.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for generation
        prompts: List of input prompts
        generation_config: Configuration for generation
        batch_size: Batch size for generation

    Returns:
        List of generated texts
    """
    # Set default generation config if not provided
    if generation_config is None:
        generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }

    results = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        # Generate
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                **generation_config
            )

        # Decode outputs
        batch_results = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        # For causal LMs, remove the prompt from the output
        if isinstance(model, AutoModelForCausalLM):
            for j, prompt in enumerate(batch_prompts):
                if batch_results[j].startswith(prompt):
                    batch_results[j] = batch_results[j][len(prompt):]

        results.extend(batch_results)

    return results


# Removed create_chosen_examples function as requested


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inference script for Gemma 2 and Qwen 2.5 models")

    # Model arguments
    parser.add_argument("--model", type=str, required=True, choices=["gemma2", "qwen2.5"],
                        help="Model type to use (gemma2 or qwen2.5)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model or model identifier from huggingface.co/models. If not provided, will use the default path for the selected model.")
    # Removed --instruct parameter as requested
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to the adapter or adapter identifier from huggingface.co/models")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="Data type for model weights")

    # Inference arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input file containing prompts in JSON format")
    parser.add_argument("--input_field", type=str, default=None,
                        help="Field in the dataset to use as input (e.g., 'text', 'original_text'). If not specified, will try common fields.")
    # Removed --raw_dataset_file parameter as requested
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the generated output")
    parser.add_argument("--use_chat", action="store_true",
                        help="Whether to use chat format for the model (applies chat template to inputs)")

    # Generation config
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    # Removed --repetition_penalty parameter as requested
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether to use sampling for generation")

    args = parser.parse_args()

    # Determine model path
    model_type = args.model
    if args.model_path is None:
        # Always use the base model since --instruct parameter is removed
        model_path = SUPPORTED_MODELS[model_type]["base"]
    else:
        model_path = args.model_path

    # For all models, use the use_chat flag to determine if it's a chat model
    if args.use_chat:
        SUPPORTED_MODELS[model_type]["is_chat_model"] = True

    print(f"Using model: {model_type} ({model_path})")

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = load_tokenizer(model_path, trust_remote_code=True)

    # Load base model
    print(f"Loading model...")
    model = load_base_model(model_path, model_type, args.dtype, trust_remote_code=True)

    # Load adapter if specified
    if args.adapter_path is not None:
        print(f"Loading adapter from {args.adapter_path}...")
        model = load_adapter(model, args.adapter_path, adapter_type="lora")

    # Prepare generation config
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        # Removed repetition_penalty parameter as requested
        "do_sample": args.do_sample
    }

    # Load input data
    print(f"Loading input file: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading input file: {e}")

    print(f"Loaded {len(input_data)} items from input file")

    # Removed raw_dataset_file processing as requested

    # Process inputs and generate outputs
    results = []

    for i, item in enumerate(input_data):
        if i % 10 == 0:
            print(f"Processing item {i+1}/{len(input_data)}...")

        # Check if a specific input field is specified
        if args.input_field is not None:
            if args.input_field in item:
                prompt = item[args.input_field]
                if args.use_chat:
                    # Convert to chat format
                    messages = [{"role": "user", "content": prompt}]
                    output = chat_completion(model, tokenizer, messages, model_type, generation_config, args.use_chat)
                else:
                    # Use as plain text
                    output = generate_text(model, tokenizer, prompt, generation_config)
            else:
                print(f"Skipping item {i+1}: field '{args.input_field}' not found")
                continue
        else:
            print(f"Skipping item {i+1}: unknown format")
            continue

        # Add to results
        results.append({
            "input": messages if "messages" in locals() else prompt,
            "output": output
        })

    # Save results
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")

    # Print sample results
    print("\nSample results:")
    for i, result in enumerate(results[:3]):
        print(f"\nInput {i+1}:\n{result['input']}")
        print(f"Output {i+1}:\n{result['output']}")

    if len(results) > 3:
        print(f"... and {len(results) - 3} more results")


if __name__ == "__main__":
    main()