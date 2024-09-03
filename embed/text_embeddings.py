import onnxruntime as ort
from inference.models import Clip
from pprint import pprint

print(f"Available Model Providers: {ort.get_device()}")
clip = Clip(model_id="clip/ViT-B-16", onnxruntime_execution_providers=["CUDAExecutionProvider"])


#image_embedding = clip.embed_image(image="drinks/bloody mary/937ab91a-24d2-430e-965e-b79b8ef5b344.jpg")
compairson =  clip.compare(subject="hello world", subject_type="text", prompt=["globe", "dog"])
pprint(compairson)
