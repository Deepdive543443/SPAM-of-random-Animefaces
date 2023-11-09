FROM python:3.8
WORKDIR /TWDE_MINI
COPY TWDE_MINI .
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install setuptools==58.3.0
RUN pip3 install Cython
RUN pip3 install opencv-python
RUN pip install flask torch
EXPOSE 8000
CMD ["python","main.py"]
