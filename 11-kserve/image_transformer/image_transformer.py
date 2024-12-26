import argparse
from kserve import Model, ModelServer, model_server, InferInput, InferRequest, logging
from typing import Dict, List
# from PIL import Image
# import torchvision.transforms as transforms
# import logging
# import io
# import base64
import kserve
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor()


# def image_transform(byte_array):
#     """converts the input image of Bytes Array into Tensor
#     Args:
#         instance (dict): The request input for image bytes.
#     Returns:
#         list: Returns converted tensor as input for predict handler with v1/v2 inference protocol.
#     """
#     image_processing = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     image = Image.open(io.BytesIO(byte_array))
#     tensor = image_processing(image).numpy()
#     return tensor

# for v1 REST predictor the preprocess handler converts to input image bytes to float tensor dict in v1 inference REST protocol format
class ImageTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str, headers: Dict[str, str] = None):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = True
        self.preprocessor = create_preprocessor('xception', target_size=(299, 299))
        self.classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

    def prepare_input(self, url: str) -> List:
        X = self.preprocessor.from_url(url)
        return X[0].tolist()

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        result = []

        for url in inputs['instances']:
            row = self.prepare_input(url)
            result.append(row)
            
        return {'instances': result}

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        result = []

        for prediction in inputs['predictions']:
            output = dict(zip(self.classes, prediction))
            result.append(output)

        return {'predictions': result} 

# for v2 gRPC predictor the preprocess handler converts the input image bytes tensor to float tensor in v2 inference protocol format
# class ImageTransformer(kserve.Model):
#     def __init__(self, name: str, predictor_host: str, protocol: str, headers: Dict[str, str] = None):
#         super().__init__(name)
#         self.predictor_host = predictor_host
#         self.protocol = protocol
#         self.ready = True

#     def preprocess(self, request: InferRequest, headers: Dict[str, str] = None) -> InferRequest:
#         input_tensors = [image_transform(instance) for instance in request.inputs[0].data]
#         input_tensors = np.asarray(input_tensors)
#         infer_inputs = [InferInput(name="INPUT__0", datatype='FP32', shape=list(input_tensors.shape),
#                                    data=input_tensors)]
#         infer_request = InferRequest(model_name=self.model_name, infer_inputs=infer_inputs)
#         return infer_request

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument("--predictor_host", help="The URL for the model predict function")
    parser.add_argument("--model_name", help="The name of the model", required=True)
    args, _ = parser.parse_known_args()

    # if args.configure_logging:
    #     logging.configure_logging(args.log_config_file)  # Configure kserve and uvicorn logger
    
    model_name = args.model_name
    host = args.predictor_host

    model = ImageTransformer(model_name, predictor_host=host)

    server = ModelServer()
    server.start(models=[model])