from io import BytesIO
from urllib import request
import numpy as np

from PIL import Image
import tflite_runtime.interpreter as tflite

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()

output_index = interpreter.get_output_details()[0]['index']
input_index = interpreter.get_input_details()[0]['index']

def predict(url):
    img = download_image(url)
    img = prepare_image(img, (200, 200))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_preds = preds[0].tolist()

    return float_preds[0]

def handler(event, context):
    url = event['url']
    pred = predict(url)
    return { 'pred': pred }