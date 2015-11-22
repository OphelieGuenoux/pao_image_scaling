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
    imageTest = np.array((imread(nomImage, flatten=1)-SuperScale.quantite_a_retirer)/SuperScale.quantite_a_diviser)
    imageDecompo = SuperScale.decouper_image(imageTest, 8)
    nom = nomImage.split("/")[-1:][0].split(".")[0]

    imsave("./tests/"+nom+"_original.png", imageTest.reshape((32,32))*SuperScale.quantite_a_diviser+SuperScale.quantite_a_retirer)

    [n, p] = np.shape(imageDecompo)
    imageReconstruite = np.zeros((n, 100))
    for i in range(n):
        imageReconstruite[i] = f([imageDecompo[i]])


    imageReconstruite = SuperScale.recomposer_image(np.around(imageReconstruite), 40, 40, 10)
    imageReconstruite = imageReconstruite * SuperScale.quantite_a_diviser + SuperScale.quantite_a_retirer

    imsave("./tests/"+nom+"_reconstruction.png", imageReconstruite)


testImage("./dataset/images_input/5.png")
testImage("./dataset/images_input/15.png")
testImage("./dataset/images_input/25.png")
testImage("./dataset/images_input/35.png")
testImage("./dataset/images_input/45.png")
testImage("./dataset/images_input/55.png")
testImage("./dataset/images_input/65.png")
testImage("./dataset/images_input/75.png")
testImage("./dataset/images_input/85.png")
