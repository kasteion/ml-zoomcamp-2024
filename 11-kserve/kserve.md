# Kserve

### 1. Overview

KServe is a framework for deploying Machine Learning models on top of kubernetes.

https://github.com/kserve/kserve

https://github.com/kserve/kserve

Samples:

https://github.com/kserve/kserve/tree/master/docs/samples/v1beta1

Write less yaml!

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```

- Kubeflow and KServe

It was part of Kubeflow

- Two-tier architecture

### 2. Running KServe locally

- Installing KServe locally with kind

```bash
# Delete the pervious cluster
kind delete cluster

# Created cluster again
kind create cluster

kubectl config get-contexts

kubectl config use-context kind-kind

curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.14/hack/quick_install.sh" | bash

istioclt x uninstall --purge

rm -rf istio-1.9.0

# export ISTIO_VERSION=1.6.2

kubectl get namespaces
kubectl get pod -n istio-system
kubectl describe pod istiod-777d5f7df9-62pp5 -n istio-system

kubectl get namespaces
kubectl get pod -n kserve
```

- Deploying an example model from documentation

Create a file named iris-example.yaml

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```

Apply

```bash
kubectl apply -f iris-example.yaml

# Because we installed kind
kubectl get inferenceservice
kubectl get isvc

# http://sklearn-iris.default.example.com
# https://<SERVICE_NAME>.<NAMESPACE>.<DOMAIN>

kubectl get service -n istio-system

kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80

kubectl get isvc

SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" "http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict" -d @./iris-input.json
```

### 3. Deploying a Scikit-Learn model with KServe

- Training the churn model with specific Scikit-Learn version

In this [file](https://github.com/kserve/kserve/blob/master/python/sklearn.Dockerfile) we can see the version of python that we are gonna use:

```
ARG PYTHON_VERSION=3.11
```

And [here](https://github.com/kserve/kserve/blob/master/python/sklearnserver/pyproject.toml) we can check the version of scikit-learn:

```
scikit-learn = "~1.5.1"
```

So we need to use a specific version of python and a specific version of scikit-learn. This is easier with conda:

```bash
conda create -n py37-scikit-learn-1.5.1 python=3.7 scikit-learn==1.5.1 pandas joblib
```

But we can use something like pyenv and then install the dependencies...

```bash
conda activate py37-scikit-learn-1.5.1

python -V

python

# import sklearn
# sklearn.__version__
```

Then we can train the model again by creating churn-train.py for the training process

```bash
python churn-train.py
```

- Deploying the churn prediction model with KServe

Now we can create `churn-service.yaml` with the deployment.

And activated them:

```bash
conda deactivate

python -m http.server

kubectl apply -f churn-service.yaml

kubectl get pods

kubectl logs churn-predictor-default-00001-deployment-57f4f87bd-lg842 kserve-container | less

kubectl logs churn-predictor-default-00001-deployment-57f4f87bd-lg842 kserve-container -- bash

# ls
# cd mnt/models
# ls
# python

## import joblib
## model = joblib.load('model.joblib')
## X = [{ 'contract': 'one_year', 'tenure': 34, 'monthlycharges': 56.95}]
## model.predict(X)
## model.predict_proba(X)
```

### 4. Deploying custom Scikit-Learn images with KServe

- Customizing the Scikit-Learn image

We can clone the kserve repo

```bash
git clone git@github.com:kserve/kserve.git

cd kserve/python

vim sklearn.Dockerfile

# We can edit the version of python
# ARG PYTHON_VERSION=3.11
# Maybe to... but I think that it should be 3.11
# ARG PYTHON_VERSION=3.12

vim sklearnserver/pyproject.toml

# We can edit the version of scikit-learn
# scikit-learn = "~1.5.1"

vim sklearnserver/sklearnserver/model.py

# Edit result = self._model.predict(instances) to use predict_proba
# But we can send an env variable to use predict_proba
# ENV_PREDICT_PROBA

docker build -t kserve-sklearnserver:3.11-1.5.1 -f sklearn.Dockerfile .
```

Now we will need to train the model again with the correct versions of python and scikit-learn.

> Goto **3. Deploying a Scikit-Learn model with KServe**

- Running KServe service locally

```bash
docker run -it --rm \
  -e ENV_PREDICT_PROBA=true
  -v "$(pwd)/model.joblib:/mnt/models/model.joblib" \
  -p 8081:8080 \
  kserve-sklearnserver:3.11-1.5.1 \
  --model_dir=/mnt/models \
  --model_name=churn

python churn-test.py
```

We can edit the churn-service.yaml (I created a copy) `churn-service-custom.yaml`

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "churn"
spec:
  predictor:
    containers:
      image: kserve-sklearnserver:3.11-1.5.1
    model:
      modelFormat:
        name: sklearn
      storageUri: "http://172.31.13.90:8000/model.joblib"
      resources:
        requests:
          cpu: 300m
          memory: 256Mi
        limits:
          cpu: 300m
          memory: 256Mi
```

Not sure if it will work but...

```bash
kubectl apply -f churn-service-custom.yaml

kubectl get pods

kubectl get isvc
```

### 5. Serving Tensorflow models with KServe

- Converting the Keras model to saved_model format

First we need to download a model

```bash
wget https://github.com/alexeygrigorev/mlbookcamp-code/releases/download/chapter7-model/xception_v4_large_08_0.894.h5
```

And we use the code in convert.py to convert the model

```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v4_large_08_0.894.h5')

tf.saved_model.save(model, 'clothing-model')
```

So we run

```bash
python convert.py

mv clothing-model
mkdir 1
mv assets/ saved_model.pb variables/ 1

zip -r clothing-mode.zip *
```

- Deploying the model

We wan create a k8s manifest clothes-service.yaml

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "clothes"
spec:
  predictor:
    tensorflow:
      storageUri: "http://172.31.13.90:8000/clothes/clothing-model/clothing-model.zip"
      resources:
        requests:
          cpu: 500m
          memory: 256Mi
        limits:
          cpu: 1000m
          memory: 512Mi
```

And apply it

```bash
kubectl apply -f clothes-service.yaml

kubectl get pod

kubectl logs clothes-predictor-default-00001-deployment-68b9778f7-8qqck kserve-container | less
```

- Preparing the input

### 6. KServe transformers

- Why we do need transformers

Because the client doesn't need to know how to transform the data for our model.

- Creating a service fro pre and post processing

https://github.com/kserve/kserve/tree/master/docs/samples/v1beta1/transformer

https://github.com/kserve/website/tree/main/docs/modelserving/v1beta1/transformer/torchserve_image_transformer

1. create image_transformer.py

2. run it

```bash
python image_transformer.py --predictor_host=localhost:8080 --model_name=clothes --http_port=8081
```

3. Port forward

```bash
kubectl get pod

kubectl port-forward clothes-predictor-default-00001-deployment-8466c887b6-fjcgp 8080:8080
```

4. We can test it locally

```bash
python test_transformer.py
```

5. Create a new clothes-service.yaml

6. Apply it

```bash
kubectl apply -f clothes-service.yaml

kubectl get pod
```

- Using existing transformers

### 7. Deploying with Kserve and EKS

- Creating an EKS cluster
- Installing KServe on EKS
- Configuring the domain
- Setting up S3 access
- Deploying the clothing model

### 8. Summary

- Less yaml, faster deployment
- Less stability
- The need for Ops is not gone

### 9. Explore more

- Helm charts
- Kubeflow, Kubeflow pipelines
- Sagemaker
- A lot of vendors that take care of Ops
