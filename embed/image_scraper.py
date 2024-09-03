from duckduckgo_search import DDGS
from fastcore.all import *
from fastai.vision.all import *
from time import sleep


def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(DDGS().images(term, max_results=max_images)).itemgot('image')

searches = 'martini', "white russian", "bloody mary"
path = Path('drinks')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} drink photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)


# Remove any failed downloads.
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f"Remoeved {len(failed)} failed image downloads")