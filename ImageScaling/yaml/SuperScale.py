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
    def initData():
        #self.class_names = ['0', '255']
        # On recupere les images pour entrainer le reseau.
        input = []
        for im in os.listdir("./dataset/images_input"):
        	image = imread("./dataset/images_input/"+im, flatten=1)
        	input.append(image.flatten())
        SuperScale.Xtrain = np.array(input[0:1500])/255 #si on met -1 1 mieux
        SuperScale.Xtrain = SuperScale.Xtrain.astype(int)
        SuperScale.Xval = np.array(input[1501:])/255
        SuperScale.Xval = SuperScale.Xval.astype(int)

        output = []
        for im in os.listdir("./dataset/images_output"):
        	image = imread("./dataset/images_output/"+im, flatten=1)
        	output.append(image.flatten())
        SuperScale.ytrain = np.array(output[0:1500])/255
        SuperScale.ytrain = SuperScale.ytrain.astype(int)
        SuperScale.yval = np.array(output[1501:])/255
        SuperScale.yval = SuperScale.yval.astype(int)

    def __init__(self,val):
        if(val=='train'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xtrain, y=SuperScale.ytrain)
        elif(val=='valid'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xval, y=SuperScale.yval)
        else:
            print('choisir entre train et valid')
