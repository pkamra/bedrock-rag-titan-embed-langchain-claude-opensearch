# Use the official Python 3.10 image as the base image
FROM python:3.10.6

# Set the working directory inside the container
WORKDIR /app

# Set the AWS region environment variable
ENV AWS_DEFAULT_REGION=us-east-1


# Copy the local files into the container
COPY ./requirements.txt /app/requirements.txt
COPY ./startup_script.py /app/startup_script.py
COPY ./utils/bedrock.py /app/utils/bedrock.py
COPY ./utils/__init__.py /app/utils/__init__.py
COPY ./files/car_manual.pdf /app/files/car_manual.pdf

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
CMD [ "python", "./startup_script.py"]