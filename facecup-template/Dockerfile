FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY nextera_test.py .

CMD [ "python","-u","./nextera_test.py" ]
