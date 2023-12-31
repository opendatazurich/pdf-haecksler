[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive)

# PDF HÄCKSLER

## Automated segmentation of City Zürich's PDF archive of architectural competitions.

### Description
This repository contains code for the automated cropping of PDF files that contain architectural <b>images</b> (pixel-based) and <b>drawings</b> (vector-based) into individual files.<br>
Extracting images from PDFs is a relatively straightforward task, but vector drawings are quite more challenging. A visually single entity (eg a floor-plan) is composed of multiple (SVG) parts without explicit information of how they group into one.

To tackle this challenge, the current approach is framing the problem as an <b>Object Detection</b> one. More specifically:
- each PDF is initially converted into a list of (one or more) images.
- each image of the PDF is fed into a fine-tuned Transformer network (DETR). The output includes two categories of information, namely the detected bounding box of each entity and its predicted class (0:image or 1:drawing).
- each PDF is cropped into multiple PDF and/or PNG files and stored in disk, based on the detected bounded boxes.

The current repository contains the code to perform the automatic segmentation, as well as the code to expose it as an API using FastAPI.

### Input/Output File Structure
With a file named `example_file.pdf` as an input, the output will be a folder looking like this:
```
example_file
├── drawings
│   ├── pdf
│   │   ├── optimized
│   │   ├── original
│   ├── png
│   ├── thumbs
├── images
│   ├── pdf
│   │   ├── optimized
│   │   ├── original
│   ├── png
└── └── thumbs
```

### Contents
- The folder `/backend` contains the code for the automated segmentation and the API
- The api is under `/backend/src/api`
- The folder `/db` is used by the backend as a postgres database for the API-related processes (user auth etc.)
- Alembic is used to bridge CRUD operations between FastAPI and the database
- The main project code is under the `/backend/src/app` dir
- The file `/backend/src/app/functions.py` contains all the segmentation functions and is called by `/backend/src/app/main.py`.
- The trained model is <b>not</b> included in the repo, but must be added manually in `/backend/model` as `backend/model/model.ckpt`.

*!!! The current version of the trained model, as well as the code configuration in this repository is working for PDF files that contain architectural drawings and renders. In the future more categories can be added (like texts or tables). This will require an updated `model.ckpt` file (which is out of the scope of this repository), as well as some minor changes in the `/backend/src/app/` files.*


### Install
1. Make sure you add the trained model as: `backend/model/model.ckpt`
2. Create a `.env` file in the root folder (follow `.env_example`)
3. Build the container (`docker-compose build`)

### Use
1. In `docker-compose.yml` <b>optionally</b> configure the following parameters:
    - `CROP_OFFSET` : percentage of offset of the detected bounding box (recommended: 0.04)
    - `THUMB_SIZE`  : pixel size for thumbnails (eg 190)
    - `THRESHOLD`   : threshold value above which the predicted bounding boxes are considered valid (recommended: ~0.75)
    - `PROCESSING_RES` : dpi value for PDF-to-image conversion (recommended: 300)
    - `COMPRESSION_LEVEL` : Ghostscript compression category (recommended: 4)
    - `COMPRESSION_WAIT` : seconds to wait until compression is skipped (eg 300 seconds) - *! compression can become really slow for certain documents so it's sometimes preferable to skip it after some processing time.*
2. Run the Docker container (`docker-compose up`).
3. Register as a user with a POST request at 127.0.0.1:8000/users/register by specifying *username* and *password*. eg:
    <br>`curl -X POST -H "Content-Type: application/json" -d '{"username": <USER>, "password": <PWD>}' http://127.0.0.1:8000/users/register`
4. Login with a POST request at 127.0.0.1:8000/users/login with your credentials and obtain Bearer Token. eg:
    <br>`curl -X POST -H "Content-Type: application/json" -d '{"username": <USER>, "password": <PWD>}' http://127.0.0.1:8000/users/login`
5. Using the token make a GET request at 127.0.0.1:8000/users/process-pdf including a PDF attachment. eg:
    <br>`curl -X GET -H "Authorization: Bearer <TOKEN>" -F file=@"<INPUT>.pdf" http://127.0.0.1:8000/process-pdf > <OUTPUT>.zip`


### Credits
- The Object Detection algorithm (DETR) was published by facebook research https://github.com/facebookresearch/detr.
- Our Object Detection model is based on <i>facebook/detr-resnet-50</i> and has been fine-tuned on custom training data.
- It is loaded as a Pytorch module, using <b>huggingface</b> https://github.com/huggingface/transformers and <b>pytorch lightning</b> https://github.com/PyTorchLightning/pytorch-lightning.
- PDF processing is using PymuPDF https://github.com/pymupdf/PyMuPDF
- PDF optimisation is based on Ghostscript https://www.ghostscript.com
