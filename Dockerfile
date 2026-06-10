# Use the official Python 3.12 image
FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
# Setting ownership to the "user"
COPY --chown=user . /app

# Install the required dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 7860, which is the default port expected by Hugging Face Spaces
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
