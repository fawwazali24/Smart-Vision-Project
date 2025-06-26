FROM python:3.10-slim

# Install required system libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-0 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all files to /app
COPY requirements.txt /app/
COPY app.py /app/
COPY fresh_model.pt /app/
COPY another_model.pt /app/
COPY model_2.pt /app/
# Copy index.html to the /app directory
COPY templates/ /app/templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade tensorflow keras gunicorn
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app
# Preload the model to avoid runtime delays

# Expose the port
EXPOSE 8080

# Command to run the app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]