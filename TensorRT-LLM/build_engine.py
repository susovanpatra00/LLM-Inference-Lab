"""
build_tensorrt_engine.py
Build TensorRT-LLM engine from Hugging Face model
"""

import os
import subprocess


def build_trt_engine(model_name: str, output_dir: str):
    """
    Build TensorRT-LLM engine
    
    This is a simplified example. Actual building process may vary.
    """
    
    print(f"Building TensorRT-LLM engine for {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Example command (adjust based on your TensorRT-LLM version)
    cmd = f"""
    python -m tensorrt_llm.commands.build \
        --model_dir {model_name} \
        --output_dir {output_dir} \
        --dtype float16 \
        --max_batch_size 8 \
        --max_input_len 512 \
        --max_output_len 512
    """
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Engine built successfully at {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Engine build failed: {e}")


if __name__ == "__main__":
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT = f"./trt_engines/{MODEL.split('/')[-1]}"
    
    build_trt_engine(MODEL, OUTPUT)