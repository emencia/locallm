FROM python:3.12.1-bookworm
COPY ./locallm /app/locallm
COPY requirements.txt /app
COPY setup.py /app
COPY setup.cfg /app
COPY MANIFEST.in /app
COPY batch_script.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install .
ENV PYTHONPATH "${PYTHONPATH}:/app"
CMD ["python3", "batch_script.py"]
