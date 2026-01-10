# Use the official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /code

# 1. Copy requirements first (for better caching)
COPY ./requirements.txt /code/requirements.txt

# 2. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 3. Copy the rest of the application code
COPY . /code

# 4. Create a non-root user (Security Best Practice for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 5. Define the command to run the app
# IMPORTANT: Hugging Face expects port 7860, not 8000!
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]