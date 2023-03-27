FROM python:3.11-slim

# set workdir
WORKDIR /code

# setup unix
RUN apt-get update
RUN apt-get install -y default-libmysqlclient-dev g++ git

# setup python
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt
ENV PYTHONPATH .

# setup app
ENV SECRETS_DIR /secrets
COPY titans/ titans/
CMD [ \
    "uvicorn", "titans.api.app:app", "--host", "0.0.0.0", "--port", "443", \
    "--ssl-keyfile=/secrets/privkey.pem", \
    "--ssl-certfile=/secrets/fullchain.pem" \
]
