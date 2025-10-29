from vllm import LLM, SamplingParams

"""
BASIC vLLM USAGE - Standard Inference Without Prefix Caching

This file demonstrates standard vLLM inference where each prompt is processed
independently. Unlike prefix_caching.py, these prompts don't share common prefixes,
so each prompt's KV cache is computed from scratch.

COMPARISON WITH PREFIX CACHING:
- Basic: Each prompt processed independently, separate KV cache computation
- Prefix Caching: Shared prefixes reuse computed KV cache blocks

vLLM ARCHITECTURE OVERVIEW:
1. PagedAttention: Uses block-based memory management for KV cache
2. Batched Processing: Efficiently processes multiple prompts together
3. Dynamic Batching: Can add/remove requests from batches dynamically
4. Memory Efficiency: Blocks can be shared when prefixes match (prefix caching)
"""

# These prompts have different starting tokens, so no prefix caching benefits
# Each prompt will have its own separate KV cache computation
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

# Sampling parameters control the generation behavior
# temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
# top_p: Nucleus sampling - only consider tokens with cumulative probability <= top_p
# max_tokens: Maximum number of new tokens to generate
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

def main():
    """
    Basic vLLM inference example.
    
    This demonstrates standard usage where each prompt is processed independently.
    No prefix caching occurs since the prompts don't share common prefixes.
    """
    
    # Initialize vLLM engine with standard configuration
    # Forcing float16 as it might get crashed and show 'EngineDeadError' because of vram constraint
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",               # Memory efficient precision, avoiding bf16 fallback
        enforce_eager=False,           # Allow CUDA graphs and optimizations
        gpu_memory_utilization=0.8,    # Reserve 80% GPU memory, prevent over-allocation on T4
        disable_log_stats=True         # Clean output without verbose logging
    )

    # Generate responses for all prompts in a batch
    # vLLM will process these efficiently but each prompt gets its own KV cache
    print("Processing independent prompts - no prefix caching benefits...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Display results
    for i, out in enumerate(outputs):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Generated: {out.outputs[0].text}")
        print("-" * 40)
    
    print("\nBasic inference characteristics:")
    print("- Each prompt processed independently")
    print("- Separate KV cache computation for each sequence")
    print("- No memory sharing between prompts")
    print("- Compare with prefix_caching.py to see the difference!")

if __name__ == "__main__":
    main()
