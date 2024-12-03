# https://gallery.ecr.aws/lambda/python
FROM public.ecr.aws/lambda/python:3.8

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install keras-image-helper
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY clothing-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.handler" ]
