import onnxruntime as ort
from inference.models import Clip
import weaviate
import weaviate.classes as wvc
from pprint import pprint

print(f"Available Model Providers: {ort.get_device()}")
clip = Clip(model_id="clip/ViT-B-16", onnxruntime_execution_providers=["CUDAExecutionProvider"])

client = weaviate.connect_to_local()
print(f"client is ready: {client.is_ready()}")

img_file = "bloody_mary.jpeg"

image_embedding = clip.embed_image(image=str(img_file))[0].tolist()

cocktails = client.collections.get("Cocktail")
response = cocktails.query.near_vector(
    near_vector=image_embedding,
    limit=2,
    return_metadata=wvc.query.MetadataQuery(certainty=True)
)

pprint(response)

client.close()