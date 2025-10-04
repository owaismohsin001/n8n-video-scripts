# # # 1. Use Debian Node base image (has apt-get)
# # FROM node:18-bullseye

# # # 2. Switch to root
# # USER root

# # # 3. Install system dependencies (Python + build tools + Tesseract OCR etc.)
# # RUN apt-get update && apt-get install -y \
# #     python3 python3-pip python3-dev \
# #     build-essential \
# #     tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara \
# #     libsm6 libxext6 libxrender-dev \
# #     ffmpeg \
# #     && rm -rf /var/lib/apt/lists/*

# # # 4. Install n8n globally with npm
# # RUN npm install -g n8n

# # # 5. Copy your project files into container
# # WORKDIR /home/node/app
# # COPY . .


# # # 6. Copy a test video into container
# # COPY input_videos/test_cut.mp4 /home/node/app/input_videos/test_cut.mp4

# # # 6. Install Python dependencies
# # RUN pip3 install --no-cache-dir -r Configuration/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# # # 7. Expose n8n default port
# # EXPOSE 5678

# # # 8. Start n8n (default command)
# # CMD ["n8n"]




# # 1. Use Debian Node base image (has apt-get)
# # Use Node base image with Debian 12 (Bookworm)
FROM node:18-bookworm

USER root


ENV OCLIF_AUTOCOMPLETIONS=false
ENV NODE_OPTIONS="--no-warnings"
ENV N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true

# Install Python + venv + build tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara \
    libsm6 libxext6 libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Make python3 point to python3.11
# Make python3 point to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1


# # 4. Install n8n globally with npm
# RUN npm install -g n8n

# RUN npm config set prefix /usr/local && npm install -g n8n
# RUN ln -sf /usr/local/bin/n8n /usr/bin/n8n
RUN npm config set prefix /usr/local && npm install -g n8n
RUN ln -sf /usr/local/bin/n8n /usr/bin/n8n

# 5. Set working directory
WORKDIR /home/node/app

# 6. Copy all project files into container
COPY . .

# 6. Copy a test video into container
COPY input_videos/test_cut.mp4 /home/node/app/input_videos/test_cut.mp4

# 7. Create a Python virtual environment
RUN python3 -m venv /opt/venv

# 8. Upgrade pip and install requirements inside the venv
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r Configuration/requirements.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu

# 9. Ensure the venv is first in PATH
# ENV PATH="/opt/venv/bin:$PATH"
ENV PATH="/usr/local/bin:/opt/venv/bin:$PATH"


# 8. Expose n8n default port
EXPOSE 5678
# Set the correct CMD to run n8n
CMD ["n8n", "start"]


# # Use Node base image with Debian 12 (Bookworm)
# FROM node:18-bookworm

# USER root

# # Install Python + venv + build tools
# RUN apt-get update && apt-get install -y \
#     python3.11 python3.11-dev python3.11-venv python3-pip \
#     build-essential \
#     tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara \
#     libsm6 libxext6 libxrender-dev \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

# # Make python3 point to python3.11
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# # Install n8n globally with npm
# RUN npm install -g n8n

# # Verify n8n installation and print version
# RUN n8n --version

# # Set working directory
# WORKDIR /home/node/app

# # Copy all project files into container
# COPY . .

# # Copy a test video into container
# COPY input_videos/test_cut.mp4 /home/node/app/input_videos/test_cut.mp4

# # Create a Python virtual environment
# RUN python3 -m venv /opt/venv

# # Upgrade pip and install requirements inside the venv
# RUN /opt/venv/bin/pip install --upgrade pip \
#     && /opt/venv/bin/pip install --no-cache-dir -r Configuration/requirements.txt \
#        --extra-index-url https://download.pytorch.org/whl/cpu

# ENV PATH="/opt/venv/bin:/home/node/app/node_modules/.bin:/usr/local/bin:$PATH"

# # Set n8n environment variables
# ENV N8N_HOST=0.0.0.0
# ENV N8N_PORT=5678

# # Expose n8n default port
# EXPOSE 5678

# # Run n8n directly from local node_modules using npx
# CMD ["npx", "n8n"] 
