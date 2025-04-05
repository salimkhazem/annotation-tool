FROM python:3.10-slim
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p checkpoints configs uploads masks annotations temp_downloads

# Copy application code
COPY . .

# Clone SAM2 repository and install it
RUN git clone https://github.com/facebookresearch/segment-anything-2.git sam2_repo && \
    cd sam2_repo && \
    pip install -e . && \
    cd .. && \
    mkdir -p checkpoints && \
    # Set up symlinks for sam2 module
    ln -s sam2_repo/sam2 /app/sam2
# Download model checkpoints
COPY download_models.sh /app/
RUN chmod +x /app/download_models.sh && \
    ./download_models.sh



# Add the SAM2 model path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port
EXPOSE 5020

# Command to run the application
CMD ["python3", "web_app.py"]
