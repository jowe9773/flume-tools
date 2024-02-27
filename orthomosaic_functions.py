#orthomosaic_functions.py

#Here is where I will store all of the functions that will be used in the automated orthomosaicing

class File_Managers:
    def __init__(self):
        print("File Managers initialized")

    def loadDn(purpose):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        directory_name = filedialog.askdirectory(title = purpose)

        return directory_name

    def loadFn(purpose):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = purpose)

        return filename
    
    def loadImageList(directory):
        import glob

        images = glob.glob(directory + '/*.jpg')
        
        return images
    
    def loadGCPsList(directory):
        import glob

        gcps = glob.glob(directory + '/*.csv')

        return gcps

    def loadCutlinesList(directory):
        import glob

        cutlines = glob.glob(directory + '/*.shp')

        return cutlines

class Image_Processing:
    def __init__(self):
        print("imageProcessing initialized")

    def georeferenceImage(input_path, output_path, gcps_fn, gcp_epsg=32615, output_epsg=32615):
        from osgeo import gdal, osr
        from pathlib import Path
        import pandas as pd

        #load gcps file into a pandas df
        gcps_df = pd.read_csv(gcps_fn)

        #create gdal gcps for each gcp in the dataframe and add them to a list
        gcps = []
        for i in range(len(gcps_df)):
            x = gcps_df.loc[i, 'x_mm']              #x of gcp
            y = gcps_df.loc[i, 'y_mm']              #y of gcp
            z = 0                                   #z of gcp (0 if N/A)
            pixel = gcps_df.loc[i, 'x_pixels']      #column of gcp in image (x)
            line = gcps_df.loc[i, 'y_pixels']       #row of gcp in image (y)

            gcp = gdal.GCP(x, y, z, pixel, line)
            
            gcps.append(gcp)

        # Open the source dataset and add GCPs to it
        src_ds = gdal.OpenShared(str(input_path), gdal.GA_ReadOnly)
        gcp_srs = osr.SpatialReference()
        gcp_srs.ImportFromEPSG(gcp_epsg)
        gcp_crs_wkt = gcp_srs.ExportToWkt()
        src_ds.SetGCPs(gcps, gcp_crs_wkt)

        # Define target SRS
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(output_epsg)
        dst_wkt = dst_srs.ExportToWkt()

        error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
        resampling = gdal.GRA_Bilinear

        # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
        tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                        None,  # src_wkt : left to default value --> will use the one from source
                                        dst_wkt,
                                        resampling,
                                        error_threshold)
        dst_xsize = tmp_ds.RasterXSize
        dst_ysize = tmp_ds.RasterYSize
        dst_gt = tmp_ds.GetGeoTransform()
        tmp_ds = None

        # Now create the true target dataset
        dst_path = str(Path(output_path).with_suffix(".tif"))
        dst_ds = gdal.GetDriverByName('GTiff').Create(dst_path, dst_xsize, dst_ysize, src_ds.RasterCount)
        dst_ds.SetProjection(dst_wkt)
        dst_ds.SetGeoTransform(dst_gt)
        dst_ds.GetRasterBand(1).SetNoDataValue(0)

        # And run the reprojection
        gdal.ReprojectImage(src_ds,
                            dst_ds,
                            None,  # src_wkt : left to default value --> will use the one from source
                            None,  # dst_wkt : left to default value --> will use the one from destination
                            resampling,
                            0,  # WarpMemoryLimit : left to default value
                            error_threshold,
                            None,  # Progress callback : could be left to None or unspecified for silent progress
                            None)  # Progress callback user data
        dst_ds = None

    def mosaic(in_dir):
        from orthomosaic_functions import File_Managers as fm
        import glob
        import subprocess

        directory = in_dir

        image_list = glob.glob(directory + "\\trimmed_cam_[1-4].tif")

        image_1 = image_list[0]

        image_2 = image_list[1]

        image_3 = image_list[2]

        image_4 = image_list[3]

        output_dn = fm.loadDn("Choose an output directory to store merged image within")


        subprocess.call(['python.exe', 'gdal_merge.py', '-o', output_dn + '\\merged.tif', '-of', 'GTiff' , image_1, image_2, image_3, image_4])

    def trim(input_fn, output_fn, cutline):
        from osgeo import gdal
        import numpy as np 

        rectified = gdal.Open(input_fn)

        gdal.Warp(output_fn, rectified, cutlineDSName = cutline, cropToCutline = True, dstNodata = np.nan)


