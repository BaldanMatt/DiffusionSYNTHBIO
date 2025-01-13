# Debian based
FROM python:3.12.8-slim-bookworm

# Set the working directory in the container
RUN mkdir /app
WORKDIR /app
RUN mkdir data/

# Copy the dependencies file to the working directory
COPY requirements.txt /app/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY src /app/src

# Command to run on container start
CMD ["bash"]
