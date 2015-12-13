from SuperScale import SuperScale

import numpy as np

var = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
[1, 2, 3, 4, 5, 6, 7, 8],
])

image = SuperScale.decouper_image(var, 4, 0)
print image
print image.shape

print "On remet ensemble"
image_recompo = SuperScale.recomposer_image(image, 8, 4, 4, 0)
print image_recompo
