FROM python:3.8
WORKDIR /AnimeDock
COPY Animestream .
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install python3-pip libopenblas-dev libopenmpi-dev libomp-dev -y
RUN pip3 install setuptools==58.3.0
RUN pip3 install Cython

RUN pip install flask gunicorn gevent torch-1.9.0a0+gitd69c22d-cp38-cp38-linux_aarch64.whl torchvision-0.10.0a0+300a8a4-cp38-cp38-linux_aarch64.whl -y
EXPOSE 5000
CMD gunicorn -w 2 app:app