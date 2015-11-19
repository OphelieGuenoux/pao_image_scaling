import theano
from pylearn2.utils import serial
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import os

from SuperScale import SuperScale

model_path = 'mlp_best.pkl'
model = serial.load( model_path )


X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )
f = theano.function( [X], Y )

#Image 4
imageTest = np.array((imread("./dataset/images_input/5.png", flatten=1)-127)/255)
imageDecompo = SuperScale.decouper_image(imageTest, 8)

#imageReconstruite = ann.fprop(theano.shared(imageTest, name='inputs')).eval()
#imageReconstruiteReshape = imageReconstruite.reshape((32, 32))
#imsave("reconstruction4.png", imageReconstruiteReshape)
imsave("original4.png", imageTest.reshape((32,32))*255+127)

[n, p] = np.shape(imageDecompo)
imageReconstruite = np.zeros((n, 100))
for i in range(n):
    imageReconstruite[i] = f([imageDecompo[i]])

imageReconstruite = SuperScale.recomposer_image(imageReconstruite, 40, 40, 10)
imageReconstruite = imageReconstruite*255+127

imsave("reconstruction4.png", imageReconstruite)
