# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file into the container and install the dependencies
COPY requirements.txt /code/
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /code
COPY . /code/

CMD ["python", "app.py", "python", "info_bot.py"]
