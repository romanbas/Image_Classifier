
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

#==================================

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#==================================

parser = argparse.ArgumentParser ()

parser.add_argument('Image_path', help = 'Image Path', type = str)
parser.add_argument('model', help = 'DL Model to use ', type = str, default='my_model.h5')
parser.add_argument('--top_k', help = 'Number of K to display', type = int, default=5 )
parser.add_argument('--category_names', help = 'List of the categories in json file', default='label_map.json')

args=parser.parse_args()

#==================================
with open(args.category_names, 'r') as f:
    class_names = json.load(f)
    
reloaded = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer})

#==================================

def process_image(P_Image):
    ts_img=tf.convert_to_tensor(P_Image, dtype=tf.float32)
    ts_img=tf.image.resize(ts_img, (224,224))
    ts_img/= 255
    np_img=ts_img.numpy()
    return np_img

def predict(Image,Model,Top_K):
    prc_img=process_image(Image)
    predict=Model.predict(prc_img)
    predict = predict[0].tolist()
    Prob, Class= tf.math.top_k(predict, k=Top_K)
    Prob=Prob.numpy().tolist()#[0]
    Class=Class.numpy().tolist()#[0]
    Labeled_Class = [class_names[str(x)] for x in Class]
    return Prob,Labeled_Class
    
#==================================

image_path = args.Image_path
img = Image.open(image_path)
test_image =np.expand_dims(np.asarray(img), axis = 0)

probs, classes = predict(test_image,reloaded, args.top_k)
print("The top K probabilities",probs)
print("The top K Classes",classes)




