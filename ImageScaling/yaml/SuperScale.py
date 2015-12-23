import theano
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

import threading
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from PIL import Image
import os

class SuperScale(DenseDesignMatrix):
    Xtrain = np.array([])
    ytrain = np.array([])
    Xval = np.array([])
    yval = np.array([])

    quantite_a_retirer_input = 127;
    quantite_a_diviser_input = 127;

    quantite_a_retirer_output = 0;
    quantite_a_diviser_output = 255;

    nbValeursTrain = 22000;
    taille_fenetre_input = 8;
    taille_fenetre_output = 10;
    recouvrement = 0;

    model_path = 'mlp_best.pkl'

    @staticmethod
    def initInput():
        input = []
        for im in os.listdir("./dataset/images_input"):
            image = imread("./dataset/images_input/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, SuperScale.taille_fenetre_input, SuperScale.recouvrement)
            input.extend(imageDecoupe)
        SuperScale.Xtrain = SuperScale.centrer_reduire( np.array( input[0:SuperScale.nbValeursTrain] ) )
        SuperScale.Xval = SuperScale.centrer_reduire( np.array( input[SuperScale.nbValeursTrain+1:]) )

    @staticmethod
    def initOutput():
        output = []
        for im in os.listdir("./dataset/images_output"):
            image = imread("./dataset/images_output/"+im, flatten=1)
            imageDecoupe = SuperScale.decouper_image(image, SuperScale.taille_fenetre_output, SuperScale.recouvrement)
            output.extend(imageDecoupe)
        SuperScale.ytrain = SuperScale.centrer_reduire( np.array( output[0:SuperScale.nbValeursTrain] ) )
        SuperScale.yval = SuperScale.centrer_reduire( np.array( output[SuperScale.nbValeursTrain+1:] ) )

    # initData permet d'initialiser les donnees d'apprentissages et de validations a partir d'images. Une fois genere on peut soit
    # choisir la matrice d'apprentissage ou de validation grace au Constructeur
    @staticmethod
    def initData():
        initInputThread = threading.Thread(target = SuperScale.initInput)
        initOutputThread = threading.Thread(target = SuperScale.initOutput)

        initOutputThread.start()
        initInputThread.start()

        initOutputThread.join()
        initInputThread.join()

        print "Donnees crees apprentissage, input: %s output: %s" % (np.array(SuperScale.Xtrain).shape, np.array(SuperScale.ytrain).shape)
        print "Donnees crees validation, input: %s output: %s" % (np.array(SuperScale.Xval).shape, np.array(SuperScale.yval).shape)
    #----------------------------------------------------------------------------------------------#



    # Fonction qui decompose une image grace a une certaine fenetre et un recouvrement
    @staticmethod
    def decouper_image(image, taille_fenetre=8, recouvrement=0.5):
        offsetX = 0
        offsetY = 0
        newImage = []
        [n, p] = image.shape

        while offsetY <= n-taille_fenetre:
            while offsetX <= p-taille_fenetre:
                fenetre = 0
                fenetre = image[offsetY:offsetY+taille_fenetre, offsetX:offsetX+taille_fenetre]
                newImage.append(fenetre.flatten())
                offsetX = offsetX + int(taille_fenetre*(1-recouvrement))
            offsetX = 0
            offsetY = offsetY + int(taille_fenetre*(1-recouvrement))
        return np.array(newImage)
    #----------------------------------------------------------------------------------------------#


    # Fonction pour recomposer une image qui a ete prealablement decoupe par une fenetre de taille fixee
    @staticmethod
    def recomposer_image(image_decomposee, taille_imageX, taille_imageY, taille_fenetre, recouvrement=0.5):
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
            offsetX = offsetX + int(taille_fenetre*(1-recouvrement))
            if offsetX+taille_fenetre > taille_imageX:
                offsetX = 0
                offsetY = offsetY + int(taille_fenetre*(1-recouvrement))

        for i in range(taille_imageY):
            for j in range(taille_imageX):
                if not len(newImage[i][j]) == 0:
                    newImage[i][j] = sum(newImage[i][j])/len(newImage[i][j])
                else:
                    newImage[i][j] = 0

        return np.array(newImage)
    #----------------------------------------------------------------------------------------------#

    @staticmethod
    def centrer_reduire(image):
        return np.array((image-SuperScale.quantite_a_retirer_input)/SuperScale.quantite_a_diviser_input)

    @staticmethod
    def centrer_reduire_inverse(image):
        return image*SuperScale.quantite_a_diviser_input+SuperScale.quantite_a_retirer_input

    @staticmethod
    def seuillage(image, seuil_bas=0.0, seuil_haut=1.0):
        [n, p] = np.shape(image)
        for i in range(n):
            for j in range(p):
                if image[i,j] < seuil_bas:
                    image[i,j] = 0.0;
                if image[i,j] > seuil_haut:
                    image[i,j] = 1.0;


    @staticmethod
    def reconstruction(image):
        model = serial.load( SuperScale.model_path )
        X = model.get_input_space().make_theano_batch()
        Y = model.fprop( X )
        f = theano.function( [X], Y )

        [n, p] = np.shape(image)
        imageReconstruite = np.zeros((n, 100))
        for i in range(n):
            imageReconstruite[i] = f([image[i]])
        return imageReconstruite

    @staticmethod
    def upscaling_traitement(image, taille_image_depart_X=32, taille_image_depart_Y=32, taille_image_finale_X=40, taille_image_finale_Y=40):
        image = SuperScale.centrer_reduire( image )
        # On decompose l'image en patchs de 8x8 avec un recouvrement de moitie
        imageDecompo = SuperScale.decouper_image(image, 8, 0.5)
        # On lance l'upscaling ici !
        imageReconstruite = SuperScale.reconstruction(imageDecompo)
        # On recompose l'image apres l'avoir agrandi a partir de patchs de 10x10 et toujours un recouvrement de moitie
        imageReconstruite = SuperScale.recomposer_image(imageReconstruite, taille_image_finale_X, taille_image_finale_Y, 10, 0.5)
        # On applique un seuillage pour eliminer d'eventuelle bruits
        SuperScale.seuillage( imageReconstruite )
        # On remet l'image dans le domaine de depart
        imageReconstruite = SuperScale.centrer_reduire_inverse( imageReconstruite )
        return imageReconstruite

    @staticmethod
    def upscaling(image_path, output_path, taille_image_depart_X=32, taille_image_depart_Y=32, taille_image_finale_X=40, taille_image_finale_Y=40):
        # On recupere le nom du fichier
        nom = image_path.split("/")[-1:][0].split(".")[0]
        # On centre et reduit l'image d'entree
        imageTest = imread(image_path, flatten=1)
        # On sauvegarde l'image original
        imsave( output_path+"/"+nom+"_original.png", imageTest.reshape(( taille_image_depart_X,  taille_image_depart_Y )) )
        imageReconstruite = SuperScale.upscaling_traitement(imageTest, taille_image_depart_X, taille_image_depart_Y, taille_image_finale_X, taille_image_finale_Y)
        # On sauvegarde l'image ainsi obtenue
        imsave( output_path+"/"+nom+"_reconstruction.png", imageReconstruite)

    @staticmethod
    def upscalingRGB(image_path, output_path, taille_image_depart_X=32, taille_image_depart_Y=32, taille_image_finale_X=40, taille_image_finale_Y=40):
        nom = image_path.split("/")[-1:][0].split(".")[0]

        imageTest = Image.open(image_path)
        if imageTest.mode == 'P':
            imageTest = imageTest.convert('RGB')
        imageTest = np.asarray(imageTest)
        imageFinal = np.zeros((taille_image_finale_X, taille_image_finale_Y, 3), dtype=np.uint8)
        for layer in range(3):
            imageFinal[..., layer] = SuperScale.upscaling_traitement(imageTest[..., layer], taille_image_depart_X, taille_image_depart_Y, taille_image_finale_X, taille_image_finale_Y)

        imsave( output_path+"/"+nom+"_original.png", imageTest )
        imsave( output_path+"/"+nom+"_reconstruction.png", imageFinal)


    # Constructeur de SuperScale, permet soit de considerer les donnees de tests ou de validations
    def __init__(self,val):
        if(val=='train'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xtrain, y=SuperScale.ytrain)
        elif(val=='valid'):
            # On appelle super pour instancer une DenseDesignMatrix avec nos donnees
            super(SuperScale, self).__init__(X=SuperScale.Xval, y=SuperScale.yval)
        else:
            print('choisir entre train et valid')
