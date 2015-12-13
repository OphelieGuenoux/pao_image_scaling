from SuperScale import SuperScale

import numpy as np
from scipy.misc import imread

var = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
])

image = SuperScale.decouper_image(var, 4)
print image
print image.shape

print "On remet ensemble"
image_recompo = SuperScale.recomposer_image(image, 8, 4, 4)
print image_recompo

input = []
image = imread("./dataset/images_input/5.png", flatten=1)
for i in range(32):
    print image[i]
for i in range(32):
    print ((image[i]-127)/127).astype(int)
imageDecoupe = SuperScale.decouper_image(image, SuperScale.taille_fenetre_input, 0.5)
input.extend(imageDecoupe)
pouet = (np.array(input)-SuperScale.quantite_a_retirer_input)/SuperScale.quantite_a_diviser_input #si on met -1 1 mieux
pouet = pouet.astype(int)
print np.array(input).shape

for i in range(16):
    print pouet[i]
print "Reconstruction"
image_recompo = SuperScale.recomposer_image(pouet, 32, 32, 8, 0.5)
for i in range(32):
    print image_recompo[i]
