import os
from PIL import Image
import numpy as np
import pandas as pd
import random

WINDOWS = ["brain_bone_window", "brain_window", "max_contrast_window", "subdural_window"]
HEM_TYPES = ["epidural", "intraparenchymal", "intraventricular", "multi", "normal", "subarachnoid", "subdural"]


def walk_paths():
    train_paths = {'batch_1': {}, 
                   'batch_2': {},
                   'batch_3': {}}
    validate_paths = {}
    test_paths = {}

    for type in HEM_TYPES:
        options = walk("brain_bone_window", type)
        full_train, test, val = divide_paths(options)

        validate_paths[type] = val
        test_paths[type] = test

        b1, b2, b3 = divide_paths(full_train, 0.33, 0.33)
        train_paths["batch_1"][type] = b1
        train_paths["batch_2"][type] = b2
        train_paths["batch_3"][type] = b3

    return train_paths, validate_paths, test_paths


def walk(window, type):
    ids = []
    for _, _, filenames in os.walk(f'dcms/{type}/{window}'):
        for filename in filenames:
            ids.append(filename)
    
    return ids


def divide_paths(options, train=0.75, test=0.15):
    # Just gathering image names so don't need to look at all windows
    num = len(options)

    random.shuffle(options)

    num_train = int(num * train)
    num_test = int(num * test)
    num_val = num - num_train - num_test

    train = options[:num_train]
    test = options[num_train:num_train+num_test]
    val = options[num_train+num_test:]

    return train, test, val


def load_batch(batch):
    for type in batch:
        ids = batch[type]
        batch[type] = {}
        for window in WINDOWS:
            batch[type][window] = load_window(ids, window, type)
    
    return batch


def load_window(img_ids, window, type):
    ids = []
    imgs = []
    for id in img_ids:
        try:
            id, img = load_image(f'dcms/{type}/{window}/{id}')
        except (Image.UnidentifiedImageError, FileNotFoundError):
            continue

        ids.append(id)
        imgs.append(img)

    data = {
        'id': ids,
        'img': imgs
    }
    
    df = pd.DataFrame(data)

    return df


def load_image(path):
    img = Image.open(path)
    img = img.resize((128, 128))

    id = os.path.split(path)[1]

    img = np.array(img)

    return (id, img)


# def to_file(name, data):
#     with open(f'{name}.json', 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)