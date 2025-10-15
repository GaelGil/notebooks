# Use NVIDIA CUDA base image for GPU
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Install Python + system deps
RUN apt-get update && apt-get install -y \
    curl git build-essential python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set workdir
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY . . 

# Install deps using uv (system-wide, reproducible if uv.lock exists)
RUN uv sync --frozen

# Expose Jupyter if you plan to run notebooks
EXPOSE 8888

# Default: start JupyterLab
CMD ["uv", "run", "jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
