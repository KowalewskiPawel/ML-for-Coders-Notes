from duckduckgo_search import DDGS
from fastdownload import download_url
from fastcore.all import *
from fastai.vision.all import *
from time import sleep

def search_images(keywords, max_images=30):
    ddgs = DDGS()
    print(f"Searching for '{keywords}'...")
    return L(ddgs.images(keywords=keywords, max_results=max_images)).itemgot('image')

dog_photo_url = search_images('dog photo', max_images=1)[0]
print(dog_photo_url)

download_url(dog_photo_url, 'dog.jpg', show_progress=True)
download_url(search_images('cat photo', max_images=1)[0], 'cat.jpg', show_progress=True)

searches = 'dog','cat'
path = Path('dog_or_cat')

for animal in searches:
    dest = (path/animal)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{animal} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{animal} photo'))
    sleep(10)
    resize_images(path/animal, max_size=400, dest=path/animal)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(f'{len(failed)} images removed')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

predicted_animal,prediction_index,prediction_probability = learn.predict(PILImage.create('dog.jpg'))
print(f"The photo depicts: {predicted_animal}. Probability: {prediction_probability[prediction_index]:.4f}")
predicted_animal,prediction_index,prediction_probability = learn.predict(PILImage.create('cat.jpg'))
print(f"The photo depicts: {predicted_animal}. Probability: {prediction_probability[prediction_index]:.4f}")