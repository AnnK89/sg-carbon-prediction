FROM tensorflow/tensorflow:2.10.0

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY interface interface
COPY ml_logic ml_logic
RUN pip install .

CMD uvicorn interface.fast:app --host 0.0.0.0 --port $PORT
