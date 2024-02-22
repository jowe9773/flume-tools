# image_sequence_from_video.py

#import packages and classes
from gp_classes import ImagePreparation
from gp_classes import Managers

#create an instance of the class
project = ImagePreparation()

#load video to sequence
video = Managers.loadFn("Choose Video to Sequence")

#choose directory to store image sequence in
image_sequence_dn = Managers.loadDn("Choose directory to store image sequence in")

#parameters for sequencing
start_time = 1
clip_duration = 0
step = 1/24


#run image sequencing function
project.extractImages(video, image_sequence_dn, start_time, clip_duration, step)
