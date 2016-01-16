import PIL
from PIL import Image
from PIL import ImageDraw
from random import randint


# donnee test avec cercle+rectangle
print("Generation donnees de tests")
for i in range(1000):
    im=Image.new("RGB", (40,40), "white")
    draw= ImageDraw.Draw(im) #zone de dessin de l'image
    draw.ellipse([randint(0,20),randint(0,20),randint(21,40),randint(21,40)], fill= (0,0,0))

    #draw.ellipse([10,20,30,25], fill=(0,0,0))
    draw.rectangle([randint(0,40),randint(0,40),randint(0,40),randint(0,40)],fill=(0,0,0))
    im.save("images_test_output/"+str(i)+".png")
    im.thumbnail((32, 32))
    im.save("images_test_input/"+str(i)+".png")
print("--> fait")

# donnee input cercle
print("Generation donnees d'apprentissages")
for j in range(1,2001,2):
    im=Image.new("RGBA", (40,40), (255,255,255))
    draw= ImageDraw.Draw(im) #zone de dessin de l'image
    draw.ellipse([randint(0,20),randint(0,20),randint(21,40),randint(21,40)], fill= (0,0,0))
    im.save("images_output/"+str(j)+".png")
    im.thumbnail((32, 32))
    im.save("images_input/"+str(j)+".png")

# donnee input rectangle
for k in range(0,2000,2) :
	img=Image.new("RGBA", (40,40), (255,255,255))
	w = randint(10, 20)
	h = randint(10, 20)
	rectangle = Image.new("RGBA", (w, h), (0,0,0))
	rectangleRot = rectangle.rotate(randint(0, 90), expand=1)
	img.paste(rectangleRot, (randint(0, 20-w), randint(0, 20-h)), rectangleRot)
	img.save("images_output/"+str(k)+".png")
	img.thumbnail((32, 32))
	img.save("images_input/"+str(k)+".png")
print("--> fait")
