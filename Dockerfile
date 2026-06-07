FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install training dependencies.
# torch and numpy are already in the base image — do not reinstall them to
# avoid overwriting the GPU-linked torch with a CPU wheel from PyPI.
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# PyTorch Geometric — must match the base image's PyTorch + CUDA version.
# Using the PyG wheel index avoids building compiled extensions from source.
RUN pip install --no-cache-dir torch-geometric \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

COPY scripts/ ./scripts/

CMD ["python", "-u", "scripts/experts.py"]
