# Use a deep learning optimized base image from NVIDIA (recommended for AWS GPU instances)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory in the container to /htr
WORKDIR /htr

# Copy your requirements.txt file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies, including MLflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# train.py will be copied to /htr/train.py
COPY train.py .
# The utils folder will be copied to /htr/utils/
COPY utils/ utils/
# run.sh will be copied to /htr/run.sh
COPY run.sh .

# Grant execute permissions to your run.sh script
RUN chmod +x run.sh

# Define the command to be executed when the container starts
ENTRYPOINT ["./run.sh"]
