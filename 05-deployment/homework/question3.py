import pickle

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

client = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([client])

probability = round(model.predict_proba(X)[0, 1], 3)

print("Probability", probability)