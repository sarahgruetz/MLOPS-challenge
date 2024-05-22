# Set the base image (for a simple deploy use the python image with your needed version)
# Use command FROM
FROM python:3.9.13

# Copy all files that you need for your workdir (your name choice)
# Use command COPY
COPY ./requirements.txt /code/requirements.txt
COPY ./server.py /code/server.py
COPY ./pricing_model.pkl /code/pricing_model.pkl

# Set your workdir (move "yourself" to your workdir to run commands inside there)
# Use command WORKDIR
WORKDIR /code

# Install your app requirements and run necessary commands
# Use command RUN
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Expose a port to your container listen
# Use command EXPOSE
EXPOSE 80

# Define the command to execute your container
# Use command CMD (only one per dockerfile)
CMD ["fastapi", "run", "server.py", "--port", "80"]
