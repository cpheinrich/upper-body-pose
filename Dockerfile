# A solid CPU-based image to start from, provided by google for deep learning.
FROM gcr.io/deeplearning-platform-release/base-cpu@sha256:4d7a2b0e4c15c7d80bf2b3f32de29fd985f3617a21384510ea3c964a7bd5cd91

# Install pip requirements first
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy over files. You may need to edit the lines below if you have other files or directories
COPY *.py src/

# Cd into src
WORKDIR src/

# Define the entrypoint to be the server.
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]