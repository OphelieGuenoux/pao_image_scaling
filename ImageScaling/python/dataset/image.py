import PIL
from PIL import Image
from PIL import ImageDraw
from random import randint


# donnee test avec cercle+rectangle
for i in range(1000):
    im=Image.new("RGB", (64,64), "white")
    draw= ImageDraw.Draw(im) #zone de dessin de l'image
    draw.ellipse([randint(0,32),randint(0,32),randint(33,64),randint(33,64)], fill= (0,0,0))

    #draw.ellipse([10,20,30,25], fill=(0,0,0))
    draw.rectangle([randint(0,64),randint(0,64),randint(0,64),randint(0,64)],fill=(0,0,0))
    im.save("images_test/"+str(i)+".png")

# donnee input cercle
for j in range(1000):
    im=Image.new("RGB", (64,64), "white")
    draw= ImageDraw.Draw(im) #zone de dessin de l'image
    draw.ellipse([randint(0,32),randint(0,32),randint(33,64),randint(33,64)], fill= (0,0,0))
    im.save("images_output/"+str(j)+".png")

# donnee input rectangle
for k in range(1000,2000) :
	img=Image.new("RGBA", (64,64), (255,255,255))
	w = randint(10, 40)
	h = randint(10, 40)
	rectangle = Image.new("RGBA", (w, h), (0,0,0))
	rectangleRot = rectangle.rotate(randint(0, 90), expand=1)
	img.paste(rectangleRot, (randint(0, 40-w), randint(0, 40-h)), rectangleRot)
	img.save("images_output/"+str(k)+".png")
