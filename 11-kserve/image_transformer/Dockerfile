FROM python:3.11.0-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

# Using --system --deploy doesn't create the virtual environment (we don't need it because we are inside a container)
RUN pipenv install --system --deploy

COPY "image_transformer.py" .

ENTRYPOINT ["python", "image_transformer.py"]