FROM python:3.7.14

WORKDIR /foodapi

COPY main.py /foodapi/main.py
COPY model_torchfile /foodapi/model_torchfile
COPY requirements.txt /foodapi/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
