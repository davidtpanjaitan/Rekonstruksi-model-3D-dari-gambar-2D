from PIL import Image
import cv2
from keras.models import Model
from keras.layers import *
import argparse
import sys
import os
import numpy as np
from .tiramisu.model import create_tiramisu

network_weights_path = os.path.join(os.path.dirname(__file__), 'models/tiramisu_coco.h5')
input_shape = (224, 224, 3)

class TiramisuEvaluator:
    def __init__(self, net_path=network_weights_path):
        self.net = self.load_network(net_path)

    def segment(self, image, width, height):
        test_img = np.array([np.asarray(image.resize((input_shape[0], input_shape[1])))])
        prediction = self.net.predict(test_img, 1)
        # variasi 1 dari output
        prediction = np.argmax(prediction, axis=-1)
        prediction[prediction != 1] = 0
        outcome = np.resize(prediction, (input_shape[0], input_shape[1])) * 255
        #valiasi 2 dari output
        # prediction = prediction[:,:,1] 
        # outcome[outcome > 0.8] = 255
        out_im = Image.fromarray((outcome).astype('uint8')).convert('L')
        return out_im.resize((width, height))

    def load_network(self, net_path):
        number_classes = 2
        img_input = Input(shape=input_shape)
        x = create_tiramisu(number_classes, img_input)
        model = Model(img_input, x)
        model.load_weights(net_path)
        return model

if __name__ == '__main__':
    img = Image.open('images/testimage0.png').convert('RGB')
    ev = TiramisuEvaluator()
    mask = ev.segment(img, img.width, img.height)
    mask.save('images/testimage0_mask.png')