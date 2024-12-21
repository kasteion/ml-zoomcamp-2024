import requests

# With Port Forwarding
# url = 'http://localhost:9696/predict'

# From k8s cluster
url = 'http://localhost:8080/predict'

data = { 'url': 'http://bit.ly/mlbookcamp-pants' }

result = requests.post(url, json=data).json()

print(result)