FROM python:3.8
WORKDIR /AnimeDock
COPY stream .
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install setuptools==58.3.0
RUN pip3 install Cython
RUN pip3 install opencv-python
RUN pip install flask torch-1.9.0a0+gitd69c22d-cp38-cp38-linux_aarch64.whl torchvision-0.10.0a0+300a8a4-cp38-cp38-linux_aarch64.whl
EXPOSE 5000
CMD ["python","main.py","--ip=0.0.0.0","--port=5000"]
#gunicorn gevent