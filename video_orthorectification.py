#video_orthorectification.py

#The goal of this code is to orthorectify a single video, then stitch the four of them together


#load neccesary packages
from orthomosaic_functions import Image_Processing

#step 1: orthorectify original video using GCPs
ortho_vid1 = Image_Processing.orthorectifyVideo()

ortho_vid2 = Image_Processing.orthorectifyVideo()
ortho_vid3 = Image_Processing.orthorectifyVideo()
ortho_vid4 = Image_Processing.orthorectifyVideo()
#step 2: cut the extent of each video (maybe can be combined with previous step?)


#step 3: stitch together the corresponding frames from each camera