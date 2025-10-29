"""
TensorRT-LLM vs vLLM Performance Comparison
===========================================
This script compares inference performance between TensorRT-LLM and vLLM
for language model inference.

Requirements:
    pip install tensorrt-llm vllm transformers torch
"""

import time
import numpy as np
from typing import List, Dict
import torch


# ============================================================================
# 1. Configuration
# ============================================================================
class Config:
    # Model configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to your model
    # For smaller/faster testing, you can use:
    # MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Generation parameters
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # Benchmark parameters
    NUM_WARMUP = 5
    NUM_ITERATIONS = 20
    
    # Test prompts
    PROMPTS = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the benefits of regular exercise?",
        "Describe the process of photosynthesis:",
        "How does machine learning differ from traditional programming?"
    ]


# ============================================================================
# 2. TensorRT-LLM Setup
# ============================================================================
def setup_tensorrt_llm(model_name: str):
    """
    Setup TensorRT-LLM engine
    Note: This requires the model to be converted to TensorRT-LLM format first
    """
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        
        print("🔧 Setting up TensorRT-LLM...")
        
        # Path to TensorRT-LLM engine (you need to build this first)
        engine_dir = f"./trt_engines/{model_name.split('/')[-1]}"
        
        # Initialize runner
        runner = ModelRunner.from_dir(
            engine_dir=engine_dir,
            rank=0,
            debug_mode=False
        )
        
        print("✅ TensorRT-LLM setup complete")
        return runner
        
    except Exception as e:
        print(f"❌ TensorRT-LLM setup failed: {e}")
        print("💡 Note: You need to build TensorRT-LLM engine first")
        return None


def run_tensorrt_llm_inference(runner, prompts: List[str], config: Config):
    """Run inference with TensorRT-LLM"""
    if runner is None:
        return None
    
    try:
        outputs = runner.generate(
            prompts,
            max_new_tokens=config.MAX_NEW_TOKENS,
            end_id=runner.tokenizer.eos_token_id,
            pad_id=runner.tokenizer.pad_token_id,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            return_dict=True
        )
        
        return outputs['output_ids']
        
    except Exception as e:
        print(f"❌ TensorRT-LLM inference failed: {e}")
        return None


# ============================================================================
# 3. vLLM Setup
# ============================================================================
def setup_vllm(model_name: str):
    """Setup vLLM engine"""
    try:
        from vllm import LLM, SamplingParams
        
        print("🔧 Setting up vLLM...")
        
        # Initialize vLLM
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on your GPU count
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=2048,  # Adjust based on your needs
        )
        
        print("✅ vLLM setup complete")
        return llm
        
    except Exception as e:
        print(f"❌ vLLM setup failed: {e}")
        return None


def run_vllm_inference(llm, prompts: List[str], config: Config):
    """Run inference with vLLM"""
    if llm is None:
        return None
    
    try:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_NEW_TOKENS,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
        
    except Exception as e:
        print(f"❌ vLLM inference failed: {e}")
        return None


# ============================================================================
# 4. Benchmark Functions
# ============================================================================
def benchmark_engine(
    engine_name: str,
    inference_func,
    prompts: List[str],
    config: Config
) -> Dict:
    """Benchmark an inference engine"""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {engine_name}")
    print(f"{'='*60}")
    
    # Warmup
    print(f"Warming up ({config.NUM_WARMUP} iterations)...")
    for _ in range(config.NUM_WARMUP):
        inference_func(prompts)
    
    # Actual benchmark
    print(f"Running benchmark ({config.NUM_ITERATIONS} iterations)...")
    latencies = []
    
    for i in range(config.NUM_ITERATIONS):
        start_time = time.time()
        outputs = inference_func(prompts)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{config.NUM_ITERATIONS} - {latency:.3f}s")
    
    # Calculate statistics
    latencies = np.array(latencies)
    results = {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }
    
    return results, outputs


def print_results(results: Dict, engine_name: str, num_prompts: int, config: Config):
    """Print benchmark results"""
    print(f"\n{'='*60}")
    print(f"{engine_name} Results")
    print(f"{'='*60}")
    print(f"Number of prompts:      {num_prompts}")
    print(f"Max new tokens:         {config.MAX_NEW_TOKENS}")
    print(f"Number of iterations:   {config.NUM_ITERATIONS}")
    print(f"\nLatency Statistics (seconds):")
    print(f"  Mean:                 {results['mean']:.4f}")
    print(f"  Std Dev:              {results['std']:.4f}")
    print(f"  Min:                  {results['min']:.4f}")
    print(f"  Max:                  {results['max']:.4f}")
    print(f"  Median (P50):         {results['p50']:.4f}")
    print(f"  P95:                  {results['p95']:.4f}")
    print(f"  P99:                  {results['p99']:.4f}")
    
    # Throughput
    tokens_per_second = (num_prompts * config.MAX_NEW_TOKENS) / results['mean']
    print(f"\nThroughput:")
    print(f"  Tokens/second:        {tokens_per_second:.2f}")
    print(f"  Requests/second:      {num_prompts / results['mean']:.2f}")


def compare_results(trt_results: Dict, vllm_results: Dict):
    """Compare results between engines"""
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    speedup = vllm_results['mean'] / trt_results['mean']
    
    print(f"\nTensorRT-LLM vs vLLM:")
    print(f"  TensorRT-LLM mean latency:  {trt_results['mean']:.4f}s")
    print(f"  vLLM mean latency:          {vllm_results['mean']:.4f}s")
    print(f"  Speedup:                    {speedup:.2f}x")
    
    if speedup > 1:
        print(f"  ✅ TensorRT-LLM is {speedup:.2f}x faster")
    else:
        print(f"  ✅ vLLM is {1/speedup:.2f}x faster")
    
    # Throughput comparison
    trt_throughput = len(Config.PROMPTS) * Config.MAX_NEW_TOKENS / trt_results['mean']
    vllm_throughput = len(Config.PROMPTS) * Config.MAX_NEW_TOKENS / vllm_results['mean']
    
    print(f"\nThroughput Comparison:")
    print(f"  TensorRT-LLM:  {trt_throughput:.2f} tokens/sec")
    print(f"  vLLM:          {vllm_throughput:.2f} tokens/sec")


# ============================================================================
# 5. Sample Output Display
# ============================================================================
def display_sample_outputs(trt_outputs, vllm_outputs, prompts: List[str]):
    """Display sample outputs from both engines"""
    print(f"\n{'='*60}")
    print("SAMPLE OUTPUTS")
    print(f"{'='*60}")
    
    if trt_outputs and vllm_outputs:
        for i, prompt in enumerate(prompts[:2]):  # Show first 2 examples
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"\nTensorRT-LLM output:")
            print(f"  {trt_outputs[i] if isinstance(trt_outputs, list) else '(output format differs)'}")
            print(f"\nvLLM output:")
            print(f"  {vllm_outputs[i]}")
            print("-" * 60)


# ============================================================================
# 6. Main Execution
# ============================================================================
def main():
    print(f"\n{'='*60}")
    print("TensorRT-LLM vs vLLM Benchmark")
    print(f"{'='*60}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Prompts: {len(Config.PROMPTS)}")
    print(f"Max new tokens: {Config.MAX_NEW_TOKENS}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires GPU.")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup engines
    trt_runner = setup_tensorrt_llm(Config.MODEL_NAME)
    vllm_engine = setup_vllm(Config.MODEL_NAME)
    
    results = {}
    
    # Benchmark TensorRT-LLM
    if trt_runner is not None:
        trt_results, trt_outputs = benchmark_engine(
            "TensorRT-LLM",
            lambda p: run_tensorrt_llm_inference(trt_runner, p, Config),
            Config.PROMPTS,
            Config
        )
        print_results(trt_results, "TensorRT-LLM", len(Config.PROMPTS), Config)
        results['trt'] = trt_results
    else:
        print("\n⚠️  Skipping TensorRT-LLM benchmark")
        trt_outputs = None
    
    # Benchmark vLLM
    if vllm_engine is not None:
        vllm_results, vllm_outputs = benchmark_engine(
            "vLLM",
            lambda p: run_vllm_inference(vllm_engine, p, Config),
            Config.PROMPTS,
            Config
        )
        print_results(vllm_results, "vLLM", len(Config.PROMPTS), Config)
        results['vllm'] = vllm_results
    else:
        print("\n⚠️  Skipping vLLM benchmark")
        vllm_outputs = None
    
    # Compare results
    if 'trt' in results and 'vllm' in results:
        compare_results(results['trt'], results['vllm'])
        display_sample_outputs(trt_outputs, vllm_outputs, Config.PROMPTS)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()