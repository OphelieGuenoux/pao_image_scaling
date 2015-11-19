import theano
print theano.config.device


import os
import pylearn2
with open("superscale.yaml", 'r') as f:
    train = f.read()

print train

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()

with open('model_final.pkl','wb') as f:
    pickle.dump(train,f)


