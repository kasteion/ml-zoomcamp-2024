import requests

request = {
    "instances": [
        "http://bit.ly/mlbookcamp-pants",
        "http://bit.ly/mlbookcamp-pants"
    ] 
}

service_name = 'clothes'
host = f'{service_name}.default.example.com'

actual_domain = 'http://localhost:8081'
service_url = f'{actual_domain}/v1/models/{service_name}:predict'

headers = {
    'Host': host
}

response = requests.post(service_url, json=request, headers=headers)

print(response)
print(response.content)
print(response.json())
