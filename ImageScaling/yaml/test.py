from SuperScale import SuperScale
import os
from scipy.misc import imread


#SuperScale.upscaling("./dataset/images_input/5.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/15.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/25.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/35.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/1456.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/1654.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/1780.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/1802.png", "./tests")
#SuperScale.upscaling("./dataset/images_input/1994.png", "./tests")

#SuperScale.upscaling("./tests/images/57-0.png", "./tests")
#SuperScale.upscaling("./tests/images/57-1.png", "./tests")
#SuperScale.upscaling("./tests/images/57-2.png", "./tests")
#SuperScale.upscaling("./tests/images/57-3.png", "./tests")
#SuperScale.upscaling("./tests/images/57-4.png", "./tests")
#SuperScale.upscaling("./tests/images/5_grosse.png", "./tests", 64, 64, 80, 80)
#SuperScale.upscalingRGB("./tests/images/oiseau.png", "./tests", 256, 256, 320, 320)
#SuperScale.upscaling("./tests/images/53.png", "./tests")
#SuperScale.upscaling("./tests/images/177.png", "./tests")
#SuperScale.upscaling("./tests/images/119.png", "./tests")
#SuperScale.upscaling("./tests/images/212.png", "./tests")
#SuperScale.upscaling("./tests/images/1241.jpg", "./tests")
#SuperScale.upscalingRGB("./tests/images/rectangleR.png","./tests")

for im in os.listdir("./dataset/images_test_input"):
	SuperScale.upscaling("./dataset/images_test_input/"+im, "./vrai_test")

i = 0
erreur = []
for im in os.listdir("./dataset/images_test_output"):
	indice = im.split(".")[0]
	imagePred = imread("./vrai_test/"+indice+"_reconstruction.png", flatten=1)
	imageVrai = imread("./dataset/images_test_output/"+im, flatten=1)
	erreur.append(sum(sum(abs(imageVrai-imagePred)))/(40*40))
	i = i + 1
mse = sum(erreur)/len(erreur)
print("Variation moyenne d'un pixel : ")
print(mse)

#roundval = 100

#test2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#test1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#test0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#testcond = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#y = f([test2])
#print y.round(roundval)
#y = f([test1])
#print y.round(roundval)
#y = f([test0])
#print y.round(roundval)
#y = f([testcond])
#print y.round(roundval)
