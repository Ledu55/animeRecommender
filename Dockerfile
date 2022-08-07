FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++
RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:$PORT wsgi

# Creating app... done, ⬢ safe-waters-36970
# https://safe-waters-36970.herokuapp.com/ | https://git.heroku.com/safe-waters-36970.git