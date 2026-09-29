import subprocess

try:
    import modal
except ImportError:
    print("Modal is not installed. Please run `pip install modal` first.")
    exit(1)

# Set up the Modal app
app = modal.App("gpt2-attention-benchmark")

# Define the environment: Python 3.11 with our dependencies and local code mounted
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "tiktoken", "matplotlib")
    .add_local_dir(".", remote_path="/root/app")
)

# Request an A100 GPU
@app.function(gpu="A100", image=image, timeout=900)
def run_benchmark_on_gpu():
    import os
    os.chdir("/root/app")
    print("Running 124M scale eager compiled benchmark on Modal NVIDIA A100 GPU...")

    cmd = [
        "python3", "-u", "benchmark.py",
        "--attention", "mha", "mqa", "gqa", "mla",
        "--seq-len", "256", "512", "1024", "2048",
        "--batch-size", "1", "8", "16", "32",
        "--dtype", "bfloat16",
        "--kernel", "eager",
        "--compile",
        "--n-warmup", "3",
        "--n-runs", "10",
        "--decode-prompt-len", "512",
        "--decode-tokens", "50",
        "--device", "cuda",
        "--output-dir", "outputs"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.wait()

    # Read the outputs to return them back to your local machine
    output_files = {}
    if os.path.exists("outputs"):
        for filename in os.listdir("outputs"):
            filepath = os.path.join("outputs", filename)
            if os.path.isfile(filepath):
                with open(filepath, "rb") as f:
                    output_files[filename] = f.read()

    return output_files

@app.local_entrypoint()
def main():
    print("Submitting job to Modal...")
    # Invoke the function
    outputs = run_benchmark_on_gpu.remote()

    # Save the files returned from the Modal container to your local disk
    import os
    os.makedirs("outputs", exist_ok=True)
    for filename, content in outputs.items():
        filepath = os.path.join("outputs", filename)
        with open(filepath, "wb") as f:
            f.write(content)

    print("Job complete! Results downloaded to local outputs/ directory.")
