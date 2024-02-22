#transfer_gopro_data.py

#imports
import glob
import shutil
from gp_classes import Managers


source_dir = Managers.loadDn("Choose source directory")

dest_dir = Managers.loadDn("Choose destination directory")

videos = glob.glob(source_dir + '/*.mp4')
for fname in videos:
    print("Copying ", fname)
    shutil.copy2(fname, dest_dir)
    