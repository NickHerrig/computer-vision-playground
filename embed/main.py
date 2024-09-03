import onnxruntime as ort
from inference.models import Clip
import weaviate
import weaviate.classes as wvc
import os
from pathlib import Path
from weaviate.exceptions import UnexpectedStatusCodeError



print(f"Available Model Providers: {ort.get_device()}")
clip = Clip(model_id="clip/ViT-B-16", onnxruntime_execution_providers=["CUDAExecutionProvider"])

client = weaviate.connect_to_local()
print(f"client is ready: {client.is_ready()}")


# Create the collection. Weaviate's autoschema feature will infer properties when importing.
try:
    cocktails = client.collections.create(
        "Cocktail",
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )
except UnexpectedStatusCodeError: 
    print("collection already setup.")


# Define the base path for drink images
drinks_path = Path('./drinks')

# List to store dictionaries for each image
drink_objs = list()

# Iterate through each drink type directory
for drink_type in os.listdir(drinks_path):
    drink_type_path = drinks_path / drink_type
    
    # Check if it's a directory
    if drink_type_path.is_dir():
        # Iterate through each image in the drink type directory
        for img_file in drink_type_path.glob('*.jpg'):

            # Generate embedding for the image
            image_embedding = clip.embed_image(image=str(img_file))[0].tolist()

            drink_objs.append(wvc.data.DataObject(
                properties={
                    'drink_type': drink_type,
                },
                vector=image_embedding
            ))

drinks = client.collections.get("Cocktail")
result = drinks.data.insert_many(drink_objs)
print(result)

client.close()