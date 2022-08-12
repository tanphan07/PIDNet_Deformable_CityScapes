FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /PIDNet

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 nano  -y
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /PIDNet/requirements.txt
RUN pip3 install -r requirements.txt