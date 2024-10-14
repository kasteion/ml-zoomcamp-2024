# Web Services: introduction to Flask

- Installing flask

```bash
pip install flask
```

- Writing a simple ping/pong app

ping.py

```python
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

```

- Querying it with `curl` and browser

```bash
curl http://0.0.0.0:9696/ping
curl http://localhost:9696/ping
```
