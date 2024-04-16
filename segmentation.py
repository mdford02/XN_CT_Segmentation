import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json
import ast
import sys

FLAGGED = []
NOT_FOUND = 1010101

with open('../hem_seg_labels/flagged.txt', 'r') as f:
    for id in f.readlines():
        FLAGGED.append(id.strip())

# ct window preferences ->
# epidural -> brain, max_contrast, brain_bone, subdural

# intraparenchymal -> brain, max_contrast, brain_bone, subdral
# intraventricular -> brain, max_contrast, brain_bone, subdural
# multi -> brain, max_contrast, brain_bone, subdural
# normal -> n/a
# subarachnoid -> brain, max_contrast, brain_bone, subdural
# subdural -> subdural, brain, max_contrast, brain_bone
EPIDURAL = INTRAPARENCHYMAL = INTRAVENTRICULAR = MULTI = SUBARACHNOID = ["brain_window", "max_contrast_window", "brain_bone_window", "subdural_window"]
SUBDURAL = ["subdural_window", "brain_window", "max_contrast_window", "brain_bone_window"]

# goal:
# read through each csv of segmentation info
# for each segment with "correct_label" > "majority_label"
# look through img data and pull first available scan from above list
# resize to 128x128 (possibly convert to greyscale)
# cast x-y coordinate values into 128x128 array 
# apply Bresenham's to draw polygons fully assuming empty array spaces exist


def generate_pixel_mask(dim, pixels):
    """
    Generate a pixel mask (2d np array of 0s, 1s) of dimension dim
    1s represent the border of the polygon determined by pixels, where pixels is
    [{'x': float, 'y': float}, ...]
    """
    mask = np.ones((dim, dim), dtype=int)
    vertices = []
    pixels = ast.literal_eval(pixels)

    if len(pixels) == 1:
        if pixels[0]:
            pixels = pixels[0]
        else:
            return NOT_FOUND
    elif len(pixels) == 2:
        if pixels[0]:
            pixels = pixels[0]
        elif pixels[1]:
            pixels = pixels[1]
        else:
            return NOT_FOUND
    elif len(pixels) == 0:
        return NOT_FOUND

    for pixel in pixels:
        x = int(float(pixel['x']) * dim)
        y = int(float(pixel['y']) * dim)
        vertices.append((x, y))

    # Add bresenham's?
    draw_polygon(mask, vertices)


    return mask


# Bresenham's impl.
def draw_line(matrix, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        # Set the pixel value to 1 (draw the edge)
        matrix[y0][x0] = 0
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def draw_polygon(matrix, vertices):
    # Draw edges between consecutive vertices
    for i in range(len(vertices)):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % len(vertices)]  # Wrap around to connect the last vertex with the first one
        draw_line(matrix, x0, y0, x1, y1)


def load_image(id, type, pref, mask_pixels):
    """
    Tries to load the image from the best available window, resizes it, cast byte color values to 0, 255
    creates a pixel mask from mask_pixels, and returns both
    """
    for window in pref:
        try:
            img = Image.open(f"../dcms/{type}/{window}/{id}")
            img = img.resize((128, 128))

            img = tf.cast(img, tf.float32) / 255.0
            
            mask = generate_pixel_mask(128, mask_pixels)
            
            if isinstance(mask, int):
                return NOT_FOUND

            # mask_img = np.array((255 * (1-mask)).astype(np.uint8))
            mask_img = tf.expand_dims(mask, axis=-1)
            # mask_img = tf.concat([mask_img] * 3, axis = -1)
            mask_img = tf.cast(mask_img, tf.float32)
            
            return img, mask_img
            
        except FileNotFoundError:
            continue

    # return a flag if nothing at all can be found
    return NOT_FOUND


def read_segmentation_labels(path):
    """
    Read in a csv of segmentation labels from path as a dataframe, drop extraneous columns and return
    """
    df = pd.read_csv(path)
    keep = ['Origin', 'Labeling State', 'Majority Label', 'Correct Label', 'Number of ROIs']
    subset_df = df[keep]

    del df
    
    drops = []
    for index, row in subset_df.iterrows():
        if row['Labeling State'] == 'In Progress' or row['Number of ROIs'] == '0':
            drops.append(index)

        elif row['Labeling State'] == 'Ready':
            subset_df.at[index, 'Label'] = row['Majority Label']

        else:
            subset_df.at[index, 'Label'] = row['Correct Label']

    subset_df.drop(drops, inplace=True)
    subset_df.drop(['Majority Label', 'Correct Label'], axis=1, inplace=True)


    return subset_df

 
paths = ['../hem_seg_labels/epidural_segments.csv', '../hem_seg_labels/intraparenchymal_segments.csv', '../hem_seg_labels/multi_segments.csv', '../hem_seg_labels/subarachnoid_segments.csv', '../hem_seg_labels/subdural_segments.csv']

# order: epidural, intrap, intrav (NO LABEL DATA), multi, subarach, subdural
labels = [read_segmentation_labels(path) for path in paths]
label_prefs = [EPIDURAL, INTRAPARENCHYMAL, MULTI, SUBARACHNOID, SUBDURAL]
types = ["epidural", "intraparenchymal", "multi", "subarachnoid", "subdural"]

x_train = []
y_train = []

x_test = []
y_test = []

for label_df, pref, type in zip(labels, label_prefs, types):
    num_rows = int(len(label_df) * 0.75)
    # Train Data
    for index, row in label_df.iloc[:num_rows].iterrows():
        if row['Origin'] not in FLAGGED:
            try:
                img, mask = load_image(row['Origin'], type, pref, row['Label'])
                x_train.append(img)
                y_train.append(mask)
            except TypeError:
                continue
    # Test Data        
    for index, row in label_df.iloc[num_rows:].iterrows():
        if row['Origin'] not in FLAGGED:
            try:
                img, mask = load_image(row['Origin'], type, pref, row['Label'])
                x_test.append(img)
                y_test.append(mask)
            except TypeError:
                continue

TRAIN_LENGTH = len(x_train)
BATCH_SIZE = 32
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

x_train = tf.data.Dataset.from_tensor_slices(x_train)
x_test = tf.data.Dataset.from_tensor_slices(x_test)

y_train = tf.data.Dataset.from_tensor_slices(y_train)
y_test = tf.data.Dataset.from_tensor_slices(y_test)

train_data = tf.data.Dataset.zip((x_train, y_train))
train_data = train_data.shuffle(buffer_size = len(x_train))
train_data = train_data.batch(BATCH_SIZE)

test_data = tf.data.Dataset.zip((x_test, y_test))
test_data = test_data.shuffle(buffer_size = len(x_test))
test_data = test_data.batch(BATCH_SIZE)

# BUFFER_SIZE = len(test_data)
# test_data = tf.data.Dataset.from_tensor_slices(test_data)
# test_data = test_data.shuffle(buffer_size=BUFFER_SIZE)
# test_data = test_data.batch(BATCH_SIZE)

print("Preprocessing and batching done")

# MODEL CREATION AND TRAINING
base = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
        ]

base_outputs = [base.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base.input, outputs=base_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3)
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
         x = up(x)
         concat = tf.keras.layers.Concatenate()
         x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
         filters=output_channels, kernel_size=3, strides=2,
         padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


OUTPUT_CLASSES = 2

model = unet_model(OUTPUT_CLASSES)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

print("Model is compiled, moving to training")

# TRAINING
EPOCHS = 10

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE // EPOCHS // 2

history = model.fit(train_data, batch_size = BATCH_SIZE,  epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1,  validation_data=test_data)

model.save('hemorrhage_segmentation')


#row1 = labels.iloc[0]

#id = row1['Origin']
#type = 'epidural'
#pixels = row1['Label']

#img, mask = load_image(id, type, EPIDURAL, pixels)

#plt.figure(figsize=(15, 15))

#title = ['Image', 'Mask']

#plt.subplot(1, 2, 1)
#plt.title(title[0])
#plt.imshow(tf.keras.utils.array_to_img(img))
#plt.axis('off')

#plt.subplot(1, 2, 2)
#plt.title(title[1])
#plt.imshow(tf.keras.utils.array_to_img(mask))
#plt.axis('off')

#plt.savefig('image.png')









