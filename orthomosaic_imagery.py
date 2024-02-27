#orthomosaic_imagery.py

#import neccesary packages and functions
import glob
from orthomosaic_functions import File_Managers as fm 
from orthomosaic_functions import Image_Processing as ip

'''
As a test, I first did this through QGIS which relies on GDAL to complete these functions (as far as I know). The workflow went as follows:
1- Register (warp) each image individually
2- Remove any Nan data
3- Merge the images from each camera together
4- Clip the data to the desired extent (floodplain extent)

Since it was successful (relatively... not perfect, but I can definitely dial it in) I would like to automate it so that I can create one 
after every run.
'''

#Load images and their respective GCPs from a folder that contains them (and only them)

#choose a directory containing images, one containing GCPS for each image and one containing the cutlines for each image
image_dir = fm.loadDn("Choose the directory containing images")

gcps_dir = "C:/Users/josie/OneDrive - UCB-O365/Research/LW Flume Experiments/Winter 2024/2024 Analysis/Orthomosaic/allgcps_by_camera" # I am not planning on moving them for a while, so this should be used
    #gcps_dir = fm.loadDn("Choose the directory containing images")

cutlines_dir = fm.loadDn("Choose the directory containing images")

#choose an output directory
out_dir = fm.loadDn("Choose an output directory for the georeferenced images")


#load the filenames for each image into a list. do the same for each set of GCPs and cutlines
image_files = fm.loadImageList(image_dir)

gcp_files = fm.loadGCPsList(gcps_dir)

cutlines_files = fm.loadCutlinesList(cutlines_dir)


for i in range(len(image_files)): #for each image

    source_image = image_files[i]

    gcps_file = gcp_files[i] #select the corresponding gcps filepath

    cutlines_file = cutlines_files[i]

    georeferenced_image = out_dir + "\\" + "cam_" + str(i+1) + ".tif" #create the output filepath for this image

    ip.georeferenceImage(source_image, georeferenced_image, gcps_file) #georeference the image

    ip.trim(georeferenced_image, cutlines_file)

ip.mosaic(out_dir) #mosaic the georeferenced images



