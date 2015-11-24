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
    quantite_a_retirer = 0;
    quantite_a_diviser = 255;
    nbValeursTrain = 65333;
    taille_fenetre_input = 8;
    taille_fenetre_output = 10;

    @staticmethod
    def initData():
        # On recupere les images pour entrainer le reseau.
        input = []
        for im in os.listdir("./dataset/images_input"):
            image = imread("./dataset/images_input/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, SuperScale.taille_fenetre_input)
            print np.array(input).shape
            input.extend(imageDecoupe)
        SuperScale.Xtrain = (np.array(input[0:SuperScale.nbValeursTrain])-SuperScale.quantite_a_retirer)/SuperScale.quantite_a_diviser #si on met -1 1 mieux
        SuperScale.Xtrain = SuperScale.Xtrain.astype(int)
        SuperScale.Xval = (np.array(input[SuperScale.nbValeursTrain+1:])-SuperScale.quantite_a_retirer)/SuperScale.quantite_a_diviser
        SuperScale.Xval = SuperScale.Xval.astype(int)

        output = []
        for im in os.listdir("./dataset/images_output"):
            image = imread("./dataset/images_output/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, SuperScale.taille_fenetre_output)
            print np.array(output).shape
            output.extend(imageDecoupe)
        SuperScale.ytrain = (np.array(output[0:SuperScale.nbValeursTrain])-SuperScale.quantite_a_retirer)/SuperScale.quantite_a_diviser
        SuperScale.ytrain = SuperScale.ytrain.astype(int)
        SuperScale.yval = (np.array(output[SuperScale.nbValeursTrain+1:])-SuperScale.quantite_a_retirer)/SuperScale.quantite_a_diviser
        SuperScale.yval = SuperScale.yval.astype(int)

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
        newImage = [[[] for i in range(taille_imageX)] for _ in range(taille_imageY)]
        [n, p] = image_decomposee.shape

        for i in range(n):
            if offsetY+taille_fenetre > taille_imageY:
                print "erreur de dimension en Y"
            for j in range(taille_fenetre):
                for k in range(taille_fenetre):
                    newImage[offsetY+j][offsetX+k].append(image_decomposee[i, j*taille_fenetre+k])
            offsetX = offsetX + taille_fenetre/2
            if offsetX+taille_fenetre > taille_imageX:
                offsetX = 0
                offsetY = offsetY + taille_fenetre/2

        for i in range(taille_imageY):
            for j in range(taille_imageX):
                if not len(newImage[i][j]) == 0:
                    newImage[i][j] = sum(newImage[i][j])/len(newImage[i][j])
                else:
                    newImage[i][j] = 0

        return np.array(newImage).astype(int)

    def __init__(self,val):
        if(val=='train'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xtrain, y=SuperScale.ytrain)
        elif(val=='valid'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xval, y=SuperScale.yval)
        else:
            print('choisir entre train et valid')
