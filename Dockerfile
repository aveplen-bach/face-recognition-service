FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt update
RUN apt install build-essential cmake pkg-config curl -y
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "unified.py"]
