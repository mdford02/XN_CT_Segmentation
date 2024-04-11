import os
from PIL import Image
import numpy as np
import json

WINDOWS = ["brain_bone_window", "brain_window", "max_contrast_window", "subdural_window"]
HEM_TYPES = ["epidural", "intraparenchymal", "intraventricular", "multi", "normal", "subarachnoid", "subdural"]


def walk_paths():
    paths = {}
    for type in HEM_TYPES:
        paths[type] = {}
        for window in WINDOWS:
            paths[type][window] = walk(window, type)
    
    return paths


def walk(window, type):
    paths = []
    for dirname, _, filenames in os.walk(f'dcms/{type}/{window}'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
    
    return paths


def load_window(paths):
    contents = {}
    for path in paths:
        id, img = load_image(path)
        contents[id] = img

    return contents


def load_image(path):
    img = Image.open(path)
    img = img.resize((128, 128))

    id = os.path.split(path)[1]

    img = np.array(img)
    res = img.tolist()

    return (id, res)


def to_file(name, data):
    with open(f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)