from vllm import LLM, SamplingParams

"""
THEORY: What is Prefix Caching?

Prefix caching is an optimization technique that stores and reuses the computed
key-value (KV) cache for common prefixes across multiple prompts. In transformer
models, each token's attention computation depends on all previous tokens in the
sequence. When multiple prompts share the same prefix, we can avoid recomputing
the KV cache for that shared portion.

HOW IT WORKS IN vLLM:

1. BLOCK-BASED STORAGE: vLLM uses PagedAttention with fixed-size blocks to store
   KV cache. When a prefix is processed, its KV cache blocks are stored and can
   be referenced by multiple sequences.

2. AUTOMATIC DETECTION: vLLM automatically detects shared prefixes when processing
   batched requests. If multiple prompts start with the same tokens, it computes
   the KV cache once and reuses it.

3. MEMORY EFFICIENCY: Instead of storing separate KV caches for each sequence,
   shared blocks are referenced by multiple sequences, reducing memory usage.

4. PERFORMANCE BENEFITS:
   - Reduced computation: Skip forward pass for cached prefix tokens
   - Lower memory usage: Shared storage for common prefixes
   - Faster time-to-first-token: Especially beneficial for long shared prefixes

EXAMPLE SCENARIO:
If you have prompts like:
- "Write a story about a brave knight who..."
- "Write a story about a brave knight and..."
The shared prefix "Write a story about a brave knight" is computed once and reused.
"""

# Simulating a moderately long prefix (not too long to overflow block size)
# This creates a 350+ token prefix that will be shared across multiple prompts
long_prefix = " ".join(["This is a long prefix sentence."] * 50)

# These prompts share the same long prefix but have different continuations
# vLLM will automatically detect this shared prefix and cache its KV states
prompts = [
    long_prefix + " Hello, my name is",
    long_prefix + " The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

def main():
    """
    Demonstrates prefix caching in vLLM with shared prefixes.
    
    The key difference from basic.py is that we use prompts with a long shared prefix.
    vLLM will automatically detect this shared prefix and cache its KV states,
    leading to:
    - Faster processing for the second prompt (reuses cached prefix)
    - Lower memory usage (shared KV cache blocks)
    - More efficient batched inference
    """
    
    # Initialize vLLM engine with prefix caching enabled by default
    # Note: enforce_eager=False allows vLLM to use CUDA graphs and optimizations
    # including automatic prefix caching detection
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",                    # Memory efficient precision
        enforce_eager=False,                # Enable CUDA graphs + prefix caching
        gpu_memory_utilization=0.8,         # Reserve memory for KV cache blocks
        disable_log_stats=True,             # Clean output for demo
    )

    # Running batched generation - vLLM will automatically detect the shared prefix
    # and cache its KV states. The first prompt computes the full KV cache,
    # the second prompt reuses the cached prefix and only computes new tokens.
    print("Processing prompts with shared prefix - vLLM will cache the common part...")
    outputs = llm.generate(prompts, sampling_params)

    # Display results - notice both prompts share the same long prefix
    for i, out in enumerate(outputs):
        print(f"Prompt {i+1}: {prompts[i][:80]}...")
        print(f"Generated: {out.outputs[0].text}")
        print("-" * 40)
    
    print("\nPrefix caching benefits:")
    print("- Reduced computation for shared prefix tokens")
    print("- Lower memory usage through shared KV cache blocks")
    print("- Faster time-to-first-token for subsequent prompts")

if __name__ == "__main__":
    main()
