# import the required packages and modules
from gp_classes import Managers
import cv2
import csv

#create empty list for points
gcps = [['gcp', 'x', 'y']]

# 1- choose image to identify gcp on
image_fn = Managers.loadFn("Select image to identify GCPs on")

image = cv2.imread(image_fn)

# 2- set a number of GCPs you want to analyze (can always exit out early by clicking "x", so I will plan to iterate like 100 times)
iterations = 9
done = 0

# 3- for each iteration open the GUI and save the GCP info to a list of GCPs
for iteration in range(iterations):
    
    # if the d button is pressed, then the loop will break
    if done == 1:
        break

    # 4- open gui to select ground control point on image
    pair, name, message, done = Managers.pointSelection(image, basic_points = True, textbox= True)

    # 5- give that gcp an id (name it) and save to a list of gcps
    gcp = []
    gcp.append(name)
    gcp.append(pair[0][0])
    gcp.append(pair[0][1])
    print("GCP = ", gcp)

    gcps.append(gcp)

# 4- export the list of gcps to a .csv file
print("GCPs = ", gcps)

output_directory = Managers.loadDn("Choose a directory to store gcp data in")
csv_file = output_directory + "\\gcps.csv"

# Writing to CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(gcps)

print(f'{csv_file} has been created successfully.')

