
import theano
from pylearn2.models import mlp
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.train_extensions.live_monitoring import LiveMonitoring

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import os

#il faut tracer les courbes d'erreur app et erreur val
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

SuperScale.initData()
ds = SuperScale('train')
monitoringDs = SuperScale('valid')

# create hidden layer with 2 nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1
hidden_layer1 = mlp.Sigmoid(layer_name='hidden1', dim=800, irange=.1, init_bias=.1)

#hidden_layer2 = mlp.Sigmoid(layer_name='hidden2', dim=1024, irange=.1, init_bias=.1)
# create Softmax output layer
#output_layer = mlp.Linear(dim=1024, layer_name='output', irange=.1)
output_layer = mlp.Sigmoid(layer_name='output', dim=1600, irange=.1, init_bias=1.)
# create Stochastic Gradient Descent trainer that runs for 400 epochs
trainer = sgd.SGD(learning_rate=.05, batch_size=10, termination_criterion=EpochCounter(10), monitoring_dataset={'app': ds, 'val': monitoringDs}) #essayer de mettre un batch size a 10 (mini batch) ou stochastic (on tire au hasard)
layers = [hidden_layer1, output_layer]
# create neural net that takes two inputs
ann = mlp.MLP(layers, nvis=1024)
trainer.setup(ann, ds)

lm = LiveMonitoring(address="127.0.0.1", req_port=5655, pub_port=5656)
#print lm.list_channels().data
# train neural net until the termination criterion is true
while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    lm.on_monitor(model=ann, dataset=monitoringDs, algorithm=trainer)
    if not trainer.continue_learning(ann):
        #lm.follow_channels(['objective'])
        break

# create a theano function that operates on the full ann

X = ann.get_input_space().make_theano_batch()
Y = ann.fprop( X )
f = theano.function( [X], Y )

#Image 4
imageTest = np.array([imread("./dataset/images_input/5.png", flatten=1).flatten()])
imageTest = imageTest.astype(int)
#imageReconstruite = ann.fprop(theano.shared(imageTest, name='inputs')).eval()
#imageReconstruiteReshape = imageReconstruite.reshape((32, 32))
#imsave("reconstruction4.png", imageReconstruiteReshape)
imsave("original4.png", imageTest.reshape((32,32)))

imageReconstruite = ann.fprop(theano.shared(imageTest, name='input')).eval()
imageReconstruiteReshape = imageReconstruite.reshape((40,40))
imsave("reconstruction4.png", imageReconstruiteReshape)

#Image 6
#imageTest = np.array([imread("test_input/6_test.png", flatten=1).flatten()])/255
#imageTest = imageTest.astype(int)
#imageReconstruite = ann.fprop(theano.shared(imageTest, name='inputs')).eval()
#imageReconstruiteReshape = imageReconstruite.reshape((32, 32))
#imsave("reconstruction6.png", imageReconstruiteReshape)
#imsave("original6.png", imageTest.reshape((32, 32)))

#imageReconstruite = f(imageTest)*255
#imageReconstruiteReshape = imageReconstruite.reshape((32, 32))
#imsave("reconstruction6.png", imageReconstruiteReshape)
