RetinaNet for lung disease detection
===

An keras implementation of RetinaNet for lung disease(Pneumonia) detection

Reference: https://github.com/fizyr/keras-retinanet

## Installation 

1. Clone this repository.
2. Ensure numpy and tensorflow is installed
3. Enter `src` directory, execute `pip install . --user`
4. Run `python setup.py build_ext --inplace` to compile Cython code first.
5. the following command should be execute under `src`

## Reproduce our result

execute `bash reproduce.sh [testing image directory] [prediction csv filepath ]` will generate two file:

* [prediction csv] : the bounding box result testing image
* submission.csv: prediction result  with run-length encoding (RLE) format for kaggle submission

It take about 10 minutes on single 2080Ti GPU to predict about 5000 images.

## Training on your own dataset

### Create two CSV:
* Annotation: 

  The CSV file with annotations should contain one annotation per line. Images with multiple bounding boxes should use one row per bounding box. Note that indexing for pixel values starts at 0. The expected format of each line is:
  ```
  path/to/image.jpg,x1,y1,x2,y2,class_name
  ```
  
  Some images may not contain any labeled objects. To add these images to the dataset as negative examples, add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:

  ```
  path/to/image.jpg,,,,,
  ```

* Class

  The class name to ID mapping file should contain one mapping per line. Each line should use the following format:

  ```
  class_name,id
  ```

### Training

* `python keras_retinanet/bin/train.py csv [annotation csvfile] [class_csvfile]`
* This script also support several arguments such as batch_size, backbone_model…]
* The model will be saved under dictory `snapshots`

## Testing on your own model

### Converting a training model to inference model

The model  saved under `snapshots` are stripped down versions  and only contains the layers necessary for training (regression and classification values),  If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

`python keras_retinanet/bin/convert_model.py snapshots/[model path] [inference model saved path]`

### Testing

* `python predict/predict.py [inference model filepath] [input image directory] [prediction csv filepath] [threshold]`
* [prediction csv] will be the bounding box result testing image

