FROM python:3.7-slim-buster

RUN apt-get update \
    && apt-get -y install poppler-utils \
    && apt-get -y install ghostscript \
    && apt-get clean \
    && apt-get autoremove

WORKDIR /app
ADD . .

RUN pip3 install -r requirements.txt
