FROM tensorflow/serving:2.18.0

COPY clothing-model /models/clothing-model/1
ENV MODEL_NAME="clothing-model"
