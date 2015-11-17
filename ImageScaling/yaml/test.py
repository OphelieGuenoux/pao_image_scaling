import theano
from pylearn2.utils import serial
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import os

model_path = 'mlp_best.pkl'
model = serial.load( model_path )


X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )
f = theano.function( [X], Y )

#Image 4
imageTest = np.array([imread("./dataset/images_input/5.png", flatten=1).flatten()])
imageTest = imageTest.astype(int)
#imageReconstruite = ann.fprop(theano.shared(imageTest, name='inputs')).eval()
#imageReconstruiteReshape = imageReconstruite.reshape((32, 32))
#imsave("reconstruction4.png", imageReconstruiteReshape)
imsave("original4.png", imageTest.reshape((32,32)))

imageReconstruite = f(imageTest)
imageReconstruiteReshape = imageReconstruite.reshape((40,40))
imsave("reconstruction4.png", imageReconstruiteReshape)
