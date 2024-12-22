# Kserve

### 1. Overview

KServe is a framework for deploying Machine Learning models on top of kubernetes.

https://github.com/kserve/kserve

https://github.com/kserve/kserve

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
- Running KServe service locally

### 5. Serving Tensorflow models with KServe

- Converting the Keras model to saved_model format
- Deploying the model
- Preparing the input

### 6. KServe transformers

- Why we do need transformers
- Creating a service fro pre and post processing
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
