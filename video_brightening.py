#video_brightening.py

# import the required library
from gp_classes import ImagePreparation


# define the alpha and beta
alpha = 5 # Contrast control
beta = 50 # Brightness control

#create an instance of the image preparation class
project = ImagePreparation()

#brighten them images!
project.brightenVideo(10, 15, step = 1/24, alpha = alpha, beta = beta)