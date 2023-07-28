# Flask SAM Server

This is a Flask-based server that provides endpoints for model selection and image masking using the [Segment-Anything project](https://github.com/facebookresearch/segment-anything).

## Setup

### Download models

```bash
cd server
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Virtual environment

Create and enter the virtual environment using:

```bash
.\initEnv.bat
.\startEnv.bat
```

### Structure

- The `server/` directory contains the code for the Flask server.
- The `client/` directory contains examples of how a client application can interact with the API. This folder contains a command line client and a python client.

## Execution

### Run the server

```bash
cd server
python server.py
```

If everything went well you should see the Flask startup message

### Run the CMD client

Edit the file with your desired request and run using:

```bash
cd client/cmd_client
.\cmd_client.bat
```

### Run the Python client

Edit the file with your desired request and run using:

```bash
cd client/python_client
python client.py
```

## Server endpoints

### GET

#### /

The root endpoint just returns a hello message. Can be used to test if the API is up and running.

#### /available-models

Returns a list of the available models

#### /current-model

Returns the model that the API has currently loaded to make predictions

#### /mask

Accepts an image as input and returns a list of bounding boxes in xywh format

### POST

#### /change-model

Accepts an integer number represting the desired model to load. Returns success or fail message.

#### /predict

Accepts an image and a point in x, y coordinates representing the desired point to segment. Returns just one bounding box.

## CORS

CORS origins are defined in  ```allowed_hosts.txt``` located in the ```server/``` directory.

```
http://localhost:5173
http://localhost:5174
http://127.0.0.1:*

other hosts...
```
