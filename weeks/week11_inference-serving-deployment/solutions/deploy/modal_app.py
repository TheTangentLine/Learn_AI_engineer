"""Modal: NOT RUN (no account, and Modal's GPU functions need one). A sketch of the SAME model server on a serverless GPU: llama-server in a container with a GPU, scaled to zero when idle.

    modal deploy modal_app.py

Cold start is the cost of scaling to zero: the container starts and the model loads before the first request is served (seconds for a 135M GGUF, minutes for a large model).
"""

import subprocess

import modal

image = modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda").pip_install(
    "fastapi[standard]"
)
models = modal.Volume.from_name("gguf-models", create_if_missing=True)
app = modal.App("order-extractor-llama", image=image)


@app.function(gpu="T4", volumes={"/models": models}, scaledown_window=120, timeout=600)
@modal.concurrent(max_inputs=8)
@modal.web_server(8080, startup_timeout=120)
def serve():
    subprocess.Popen(
        [
            "/app/llama-server",
            "-m",
            "/models/order-extractor-Q8_0.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "-c",
            "4096",
            "-np",
            "8",
            "-ngl",
            "99",
        ]
    )
