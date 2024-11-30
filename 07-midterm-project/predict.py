import pickle

from flask import Flask, request, jsonify

dv_file = 'dv.bin'
model_file = 'model_d=20s=5.bin'

with open(dv_file, 'rb') as f:
    dv = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)


app = Flask('fraud_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_prob = model.predict_proba(X)[0, 1]
    y_pred = model.predict(X)


    result = {
        'prediction': float(y_pred),
        'probability': float(round(y_prob, 3)),
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)