# Fraud Detection in Energy Consumption

In industries with high volumes of transactional data, detecting anomalies such as fraud is a critical task for maintaining operational integrity and reducing financial losses. The goal of this model is to identify fraudulent activity among clients based on historical data, including demographic information and usage patterns.

The dataset comprises two key components:

1. **Client Data**: This dataset contains general information about the clients, such as their unique IDs, categories, regional and district assignments, and the date when they were onboarded. This data also includes a binary target column that indicates whether a client has been flagged for fraud in the past.

2. **Invoice Data**: This dataset records detailed transactional information for each client. It includes various consumption levels, tariff types, and other metrics related to utility usage, such as meter readings, consumption coefficients, and the duration of billing cycles.

These datasets are complementary: client data provides high-level demographic information, while invoice data captures granular details of client behavior over time.

The primary challenge is to predict and identify fraudulent clients based on patterns in their behavior and transaction history. Fraudulent behavior often manifests in specific patterns, such as:

1. Abnormally high or low consumption levels.
2. Frequent discrepancies in meter readings.
3. Multiple tariff types or unusual invoice configurations for the same client.
4. The location of the client could also be an important factor.

However, distinguishing fraudulent clients from legitimate ones is non-trivial due to:

1. **Data Complexity**: The invoice dataset contains multiple interdependent features that need to be aggregated and interpreted in the context of client-level data.
2. **Data Imbalance**: Fraud cases typically represent a small fraction of the overall dataset, which poses challenges for model training and evaluation.

This solution will be used by operational teams to:

1. **Prioritize Investigations**: Identify high-risk clients who need further review or monitoring.
2. **Improve Resource Allocation**: Focus efforts on clients most likely to be fraudulent, optimizing the use of investigative resources.

# Dataset

The dataset can be found in [data/archive.zip](data/archive.zip)

Or in [kaggle](https://www.kaggle.com/datasets/mrmorj/fraud-detection-in-electricity-and-gas-consumption?select=client_train.csv)

# Installation and Setup

This project uses Pipenv for dependency management and virtual environment creation. Follow these steps to set up the environment and install the required dependencies:

## Prerequisites

1. Ensure you have Python 3.7 or later installed.

```bash
python --version
```

2. Install Pipenv if it's not already installed:

```bash
pip install pipenv
```

3. Install the dependencies

```bash
pipenv install
```

4. Activate the virtual environment

```bash
pipenv shell
```

5. Unzip data/archive.zip

```bash
unzip data/archive.zip
```

6. Run the train script

```bash
python3 train.py
```

7. Run the prection service

```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

8. Deactivating the environment

```bash
exit
```

# Docker

1. Build the Docker image

```bash
docker build -t ml-zoomcamp-midterm .
```

2. Run the Docker image

```bash
docker run -p 9696:9696 -d ml-zoomcamp-midterm
```

# Deployment

1. Install aws elastic beanstalk cli

```bash
pipenv install awsebcli
```

2. Initialize elastic beanstalsk application

```bash
eb init -p docker -r us-east-1 fraud-service
```

3. Create elastic beanstalk environment

```bash
eb create fraud-service-dev
```

4. To terminate the elastic beanstalk environment run

```bash
eb terminate fraud-service-dev
```

## URL for testing

> http://fraud-service-dev.eba-enrjx8jr.us-east-1.elasticbeanstalk.com/predict

Example code for testing the deployment:

```python
import requests

url = "http://fraud-service-dev.eba-enrjx8jr.us-east-1.elasticbeanstalk.com/predict"
client = {
 'client_catg':  'client_catg_11',
 'region': 'region_103',
 'district': 'district_69',
 'age_months': 196,
 'consumption_level_1_mean': 934.4594594594595,
 'consumption_level_1_std': 302.84719236262026,
 'consumption_level_1_min': 165.0,
 'consumption_level_1_max': 2090.0,
 'consumption_level_2_mean': 79.97297297297297,
 'consumption_level_2_std': 125.50375967951595,
 'consumption_level_2_max': 400.0
}
requests.post(url, json=client).json()
```
