from keras.models import model_from_json
# from imutils.video import WebcamVideoStream
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from skimage import io
import cv2
import random
import tensorflow as tf
from tensorflow import Graph
import pickle
from keras import backend as K
import zipfile
import time
import spshands



json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("modelweights.h5")
# print("Loaded Model from disk")
#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

graph = tf.compat.v1.get_default_graph()


shape_to_label = {'Rock':np.array([1.,0.,0.]),'Paper':np.array([0.,1.,0.]),'Scissors':np.array([0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}  
color_dict={'Rock':(0,255,0),'Paper':(0,0,255),'Scissors':(255,0,0)}


def play(frame):

        frame = cv2.flip(frame, 1, 1)

        prepimg=cv2.resize(frame,(300,300)).reshape(1,300,300,3)
        # with graph.as_default():
        loaded_model.predict([prepimg])
        options = ['Rock','Paper','Scissors']
        NUM_ROUNDS = 3
        bplay = ""

        pred = arr_to_shape[np.argmax(loaded_model.predict(prepimg))]
        bplay = random.choice(options)            
        # print(pred,bplay)
        
        return pred,bplay

   