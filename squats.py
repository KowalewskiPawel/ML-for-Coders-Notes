from duckduckgo_search import DDGS
from fastcore.all import *

def search_images(keywords, max_images=30):
    ddgs = DDGS()
    print(f"Searching for '{keywords}'...")
    return L(ddgs.images(keywords=keywords, max_results=max_images)).itemgot('image')

urls = search_images('squat photo', max_images=1)
urls[0]

from fastdownload import download_url
dest = 'in-squat.jpg'
in_squat_url = urls[0]
download_url(in_squat_url, dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)

download_url(search_images('standing photo', max_images=1)[0], 'standing.jpg', show_progress=False)
Image.open('standing.jpg').to_thumb(256,256)

searches = 'squat','standing'
path = Path('squat_or_standing')
from time import sleep

# for o in searches:
#     dest = (path/o)
#     dest.mkdir(exist_ok=True, parents=True)
#     download_images(dest, urls=search_images(f'{o} photo'))
#     sleep(10)  # Pause between searches to avoid over-loading server
#     download_images(dest, urls=search_images(f'in {o} photo'))
#     sleep(10)
#     resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)

is_squat,_,probs = learn.predict(PILImage.create('in-squat.jpg'))
print(f"This is a: {is_squat}.")
print(f"Probability it's a squat: {probs[0]:.4f}")