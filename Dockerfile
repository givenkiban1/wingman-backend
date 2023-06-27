# Instead of creating an image from scratch, we use this image which has python installed.
FROM python:3.10.6-buster


# COPY allows you to select the folders and files to include in your docker image
# Here, we will include our api_folder and the requiremenets.txt file
COPY wingman_api /wingman_api
COPY requirements.txt /requirements.txt


# RUN allows you to run terminal commands when your image gets created
# Here, we upgrade pip and install the libraries in our requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt



# CMD controls the functionality of the container
# Here, we use uvicorn to control the web server ports

# local
CMD uvicorn wingman_api.api.wing:api --host 0.0.0.0

# deploy to gcp
# CMD uvicorn wingmanwingman_api.api.wing:api --host 0.0.0.0 --port $PORT
