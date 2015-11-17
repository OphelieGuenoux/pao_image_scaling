#!/bin/pyhton

import PIL
from PIL import Image
from PIL import ImageDraw
from random import randint
import ImageOps

for i in range(1000) :
	img=Image.new("RGBA", (64,64), (255,255,255))
	w = randint(10, 40)
	h = randint(10, 40)
	rectangle = Image.new("RGBA", (w, h), (0,0,0))
	rectangleRot = rectangle.rotate(randint(0, 90), expand=1)
	img.paste(rectangleRot, (randint(0, 40-w), randint(0, 40-h)), rectangleRot)
	img.save("output/"+str(i)+"_test.png")
