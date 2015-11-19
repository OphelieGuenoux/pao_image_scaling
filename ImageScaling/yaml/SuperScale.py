import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.train_extensions.live_monitoring import LiveMonitoring

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import os


class SuperScale(DenseDesignMatrix):
    Xtrain = np.array([])
    ytrain = np.array([])
    Xval = np.array([])
    yval = np.array([])

    @staticmethod
    def initData(taille_fenetre=4):
        #self.class_names = ['0', '255']
        # On recupere les images pour entrainer le reseau.
        input = []
        for im in os.listdir("./dataset/images_input"):
            image = imread("./dataset/images_input/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, 8)#taille_fenetre)
            print "input :"
            print np.array(input).shape
            input.extend(imageDecoupe)
        SuperScale.Xtrain = (np.array(input[0:1500])-127)/255 #si on met -1 1 mieux
        #SuperScale.Xtrain = SuperScale.Xtrain.astype(int)
        SuperScale.Xval = (np.array(input[1501:])-127)/255
        #SuperScale.Xval = SuperScale.Xval.astype(int)

        output = []
        for im in os.listdir("./dataset/images_output"):
            image = imread("./dataset/images_output/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, 10)#taille_fenetre+1)
            print "output :"
            print np.array(output).shape
            output.extend(imageDecoupe)
        SuperScale.ytrain = (np.array(output[0:1500])-127)/255
        #SuperScale.ytrain = SuperScale.ytrain.astype(int)
        SuperScale.yval = (np.array(output[1501:])-127)/255
        #SuperScale.yval = SuperScale.yval.astype(int)

        print "Donnees crees, input: %s output: %s" % (np.array(input).shape, np.array(output).shape)

    # Fonction qui decompose une image grace a une certaine fenetre
    @staticmethod
    def decouper_image(image, taille_fenetre=8):
        offsetX = 0
        offsetY = 0
        newImage = []
        [n, p] = image.shape

        while offsetY <= n-taille_fenetre:
            while offsetX <= p-taille_fenetre:
                fenetre = 0
                fenetre = image[offsetY:offsetY+taille_fenetre, offsetX:offsetX+taille_fenetre]
                newImage.append(fenetre.flatten())
                offsetX = offsetX + taille_fenetre/2
            offsetX = 0
            offsetY = offsetY + taille_fenetre/2
        return np.array(newImage)

    @staticmethod
    # Fonction pour recomposer une image qui a ete prealablement decoupe par une fenetre de taille fixee
    def recomposer_image(image_decomposee, taille_imageX, taille_imageY, taille_fenetre):
        offsetX = 0
        offsetY = 0
        newImage = np.zeros((taille_imageY, taille_imageX))
        [n, p] = image_decomposee.shape

        for i in range(n):
            if offsetY+taille_fenetre > taille_imageY:
                print "erreur de dimension en Y"
            for j in range(taille_fenetre):
                for k in range(taille_fenetre):
                    newImage[offsetY+j, offsetX+k] = ( image_decomposee[i, j*taille_fenetre+k] )
            offsetX = offsetX + taille_fenetre/2
            if offsetX+taille_fenetre > taille_imageX:
                offsetX = 0
                offsetY = offsetY + taille_fenetre/2

        return np.array(newImage)

    def __init__(self,val):
        if(val=='train'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xtrain, y=SuperScale.ytrain)
        elif(val=='valid'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xval, y=SuperScale.yval)
        else:
            print('choisir entre train et valid')
