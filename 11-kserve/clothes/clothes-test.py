import requests
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299, 299))

url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)

request = {
    "instances": X.tolist()
}

service_name = 'clothes'
host = f'{service_name}.default.example.com'

actual_domain = 'http://localhost:8081'
service_url = f'{actual_domain}/v1/models/{service_name}:predict'

headers = {
    'Host': host
}

response = requests.post(service_url, json=request, headers=headers)

predictions = response['predictions']

# print(response.json())

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

pred = predictions[0]

print(dict(zip(classes, pred)))