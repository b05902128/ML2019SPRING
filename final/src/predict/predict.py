#!/usr/bin/env python
# coding: utf-8

# In[96]:


# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import sys
import os
import numpy as np
import csv


# Constant
model_filepath = sys.argv[1]
image_dir = sys.argv[2]
output_filepath = sys.argv[3]
threshold = float(sys.argv[4])

print("model:", model_filepath)
print("input image:", image_dir)
print("output:", output_filepath)
print("threshold:", threshold)

# load retinanet model
model = models.load_model(model_filepath, backbone_name='resnet101')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Pneumonia'}




# Output csv
image_filenames=sorted(os.listdir(image_dir))

with open(output_filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['patientId', 'x', 'y', 'width', 'height', 'Target'])
    
    for i in range(len(image_filenames)):
        print( "loading imag {}/{}".format(i, len(image_filenames)), end='\r')
        image_filepath = os.path.join("..", "data", image_dir, image_filenames[i])
        image = read_image_bgr(image_filepath)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        
        records = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < threshold:
                break
            records.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
        if len(records) == 0:
            writer.writerow([image_filenames[i], '', '', '', '', '0'])
        else:
            for record in records:
                writer.writerow([image_filenames[i], record[0], record[1], record[2], record[3], '1'])
