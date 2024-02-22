import cv2
import numpy as np
import pandas as pd
from gp_classes import Managers
import pprint

# Read your images
image1 = cv2.imread(Managers.loadFn("Camera 1"))
'''image2 = cv2.imread(Managers.loadFn("Camera 2"))
image3 = cv2.imread(Managers.loadFn("Camera 3"))
image4 = cv2.imread(Managers.loadFn("Camera 4"))
'''
#Read in your ground control point files
image_1_fp = Managers.loadFn("GCPs for camera 1")
df_1 = pd.read_csv(image_1_fp)

'''image_2_fp = Managers.loadFn("GCPs for camera 2")
df_2 = pd.read_csv(image_2_fp)

image_3_fp = Managers.loadFn("GCPs for camera 3")
df_3 = pd.read_csv(image_3_fp)

image_4_fp = Managers.loadFn("GCPs for camera 4")
df_4 = pd.read_csv(image_4_fp)'''

# Define ground control points (pixel coordinates) for each image
def points(df):

    gcp_image = []
    real_world_coordinates_image = []

    for row in range(4):
        x_pixels = df.loc[row, "x_pixels"]
        y_pixels = df.loc[row, "y_pixels"]
        x_mm = df.loc[row, "x_mm"]
        y_mm = df.loc[row, "y_mm"]

        gcp_image.append([x_pixels, y_pixels])
        real_world_coordinates_image.append([x_mm, y_mm])

    gcp = np.array(gcp_image, dtype = np.float32)
    real_world_coordinates = np.array(real_world_coordinates_image, dtype = np.float32)

    return gcp, real_world_coordinates

gcp_image1, real_world_coordinates_image1 = points(df_1)

print(gcp_image1)
'''gcp_image2, real_world_coordinates_image2 = points(df_2)
gcp_image3, real_world_coordinates_image3 = points(df_3)
gcp_image4, real_world_coordinates_image4 = points(df_4)'''

# Check types and shapes
print("Type of gcp_image1:", type(gcp_image1))
print("Shape of gcp_image1:", gcp_image1.shape)
print("Type of real_world_coordinates_image1:", type(real_world_coordinates_image1))
print("Shape of real_world_coordinates_image1:", real_world_coordinates_image1.shape)

# Calculate perspective transformation matrices for each image
transformation_matrix_image1 = cv2.getPerspectiveTransform(gcp_image1, real_world_coordinates_image1)
'''transformation_matrix_image2 = cv2.getPerspectiveTransform(gcp_image2, real_world_coordinates_image2)
transformation_matrix_image3 = cv2.getPerspectiveTransform(gcp_image3, real_world_coordinates_image3)
transformation_matrix_image4 = cv2.getPerspectiveTransform(gcp_image4, real_world_coordinates_image4)'''

# Warp each image using its corresponding transformation matrix
result_image1 = cv2.warpPerspective(image1, transformation_matrix_image1, (3000, 3000))
cv2.imshow('Stitched Image', result_image1)
cv2.waitKey(0)
'''result_image2 = cv2.warpPerspective(image2, transformation_matrix_image2, (1440, 1820))
result_image3 = cv2.warpPerspective(image3, transformation_matrix_image3, (width, height))
result_image4 = cv2.warpPerspective(image4, transformation_matrix_image4, (width, height))

# Stitch the warped images together
final_result = np.zeros_like(result_image1)
final_result += result_image1
final_result += result_image2
final_result += result_image3
final_result += result_image4

# Display or save the final result
cv2.imshow('Stitched Image', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()'''