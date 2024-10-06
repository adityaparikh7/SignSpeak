# docker file to run the src folder

# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /src/
# Copy the current directory contents into the container at /app
COPY . /src/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

# Build the image
# docker build -t friendlyhello .
# Run the container
# docker run -p 4000:80 friendlyhello
# docker run -d -p 4000:80 friendlyhello