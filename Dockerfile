FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /htr

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY train.py .
COPY dict_alph .
COPY utils/ utils/
COPY model/model/
COPY run.sh .

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
