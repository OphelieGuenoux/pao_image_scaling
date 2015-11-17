import numpy as np

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


var = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
])

image = decouper_image(var, 4)
print image
print image.shape

print "On remet ensemble"
image_recompo = recomposer_image(image, 8, 4, 4)
print image_recompo
