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

def testImage(nomImage):
    imageTest = np.array((imread(nomImage, flatten=1)-SuperScale.quantite_a_retirer_input)/SuperScale.quantite_a_diviser_input)
    imageDecompo = SuperScale.decouper_image(imageTest, 8, 0)

    nom = nomImage.split("/")[-1:][0].split(".")[0]

    imsave("./tests/"+nom+"_original.png", imageTest.reshape((32,32))*SuperScale.quantite_a_diviser_input+SuperScale.quantite_a_retirer_input)

    [n, p] = np.shape(imageDecompo)
    imageReconstruite = np.zeros((n, 100))
    for i in range(n):
        imageReconstruite[i] = f([imageDecompo[i]])

    imageReconstruite = SuperScale.recomposer_image(np.around(imageReconstruite), 40, 40, 10, 0)
    imageReconstruite = imageReconstruite * SuperScale.quantite_a_diviser_output + SuperScale.quantite_a_retirer_output

    imsave("./tests/"+nom+"_reconstruction.png", imageReconstruite)


testImage("./dataset/images_input/5.png")
testImage("./dataset/images_input/15.png")
testImage("./dataset/images_input/25.png")
testImage("./dataset/images_input/35.png")
testImage("./dataset/images_input/1456.png")
testImage("./dataset/images_input/1654.png")
testImage("./dataset/images_input/1780.png")
testImage("./dataset/images_input/1802.png")
testImage("./dataset/images_input/1994.png")

test2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
test1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
test0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testcond = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y = f([test2])
print y
y = f([test1])
print y
y = f([test0])
print y
y = f([testcond])
print y
