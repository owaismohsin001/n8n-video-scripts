# Use the official n8n image
FROM n8nio/n8n:latest

# Switch to root to install packages
USER root

# Install Python 3 and required dependencies
RUN apk add --no-cache \
    python3 py3-pip \
    build-base \
    gfortran \
    bash \
    linux-headers \
    libffi-dev \
    jpeg-dev \
    zlib-dev \
    freetype-dev \
    lcms2-dev \
    tiff-dev \
    tk-dev \
    tcl-dev \
    harfbuzz-dev \
    fribidi-dev \
    libpng-dev

# Install Python 3 development headers and pkgconfig
RUN apk add --no-cache \
    python3-dev \
    pkgconfig

# Copy your Python scripts and requirements.txt
COPY *.py /scripts/
COPY requirements.txt /scripts/requirements.txt

# Install Python dependencies in a virtual environment
RUN python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir -r /scripts/requirements.txt

# Switch back to the n8n user
USER node

# Set PATH so the virtualenv is used
ENV PATH="/venv/bin:$PATH"
