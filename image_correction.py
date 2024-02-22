# image_correction.py

#load packages and classes
from gp_classes import ImagePreparation
from gp_classes import Managers

#create an instance of the Image preparation class
project = ImagePreparation()

#choose directory of files to defish
images_dn = Managers.loadDn("Choose directory of images to defish")

#choose a plave to put the defished images
defished_images = Managers.loadDn("Choose directory to store defished images")

#load lens calibration parameters
cal_params_fn = Managers.loadFn("Select lens calibration parameters file")
project.loadCalibrationParams(cal_params_fn)

##load control point data
#flume_control_points_fn = Managers.loadFn("Select FLUME control points file") #FLUME 
#project.loadFlumeControlPoints(flume_control_points_fn)

#image_control_points_fn = Managers.loadFn("Select IMAGE control points file") #IMAGE
#project.loadImageControlPoints(image_control_points_fn)

#defish the images
project.defish(imagestocorrect = images_dn, corrected_images_DN= defished_images)

#transform the images
#save_as_dn = Managers.loadDn("Choose a directory to save corrected images in")
#project.transform(save_as_dn)
