FROM python:3.12

WORKDIR /paparazzi

COPY src .
COPY run.sh .
COPY requirements.txt .
RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt


EXPOSE 8080
CMD [ "bash", "run.sh" ]