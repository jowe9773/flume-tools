# import the required library
from gp_classes import Managers
from gp_classes import ImagePreparation

#Choose image file to brighten
image_dn = Managers.loadDn("Choose Directory of Images to Brighten")

#Choose folder to store brightened images in
save_as_dn = Managers.loadDn("Choose Directory to store brightened images in")

# define the alpha and beta
alpha = 5 # Contrast control
beta = 100 # Brightness control

#create an instance of the image preparation class
project = ImagePreparation()

#brighten them images!
project.brightenImages(image_dn, save_as_dn)