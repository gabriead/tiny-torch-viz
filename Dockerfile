# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /code

# 1. Install dependencies
# We copy requirements first to cache them effectively
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 2. Copy the rest of the application
COPY . /code

# 3. Security: Create a non-root user (Required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 4. Run the application
# IMPORTANT: Hugging Face listens on port 7860 by default!
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]