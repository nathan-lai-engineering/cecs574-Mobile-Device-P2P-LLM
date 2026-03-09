FROM python:3.10-slim

# Install build dependencies for the C++ inference engine
RUN apt-get update && apt-get install -y \
    build-essential cmake git libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "numpy<2" torch==2.0.1 torchvision==0.15.2

# Install Python dependencies first (better for Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "transformers==4.33.3"

# Patch LLaMA for GQA support + torch.fx symbolic tracing compatibility
COPY patch_llama.py .
RUN python patch_llama.py

RUN apt-get update && apt-get install -y patchelf
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime/capi/ -name "*.so" -exec patchelf --clear-execstack {} +

# Copy the rest of the code
COPY . .

# Build the C++ backend components
#RUN mkdir build && cd build && cmake .. && make -j$(nproc)

RUN find /usr/local/lib/python3.10/site-packages/onnxruntime -name "*.so" -exec execstack -c {} + || true

# Default entrypoint
ENTRYPOINT ["/bin/bash"]