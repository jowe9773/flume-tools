#GoPro_classes.py

class Managers:
    def __init__(self):
        print("initialized")

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

    
    def pointSelection(image, trajectory = False, close_piece_option = False, correction_check = False, basic_points = False, textbox = False):
        import matplotlib.pyplot as plt
        import cv2
        from matplotlib.widgets import TextBox
        from matplotlib.gridspec import GridSpec
        import numpy as np

        image_final = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        fig = plt.figure(layout = 'constrained', figsize=(10,5))
        gs = GridSpec(5, 1, figure = fig)
        ax1 = fig.add_subplot(gs[0:3])
        ax1.imshow(image_final)

        pair = []
        message = 0
        done = 0
        


        #add textbox 
        text_list = []
        def on_submit(text):
            text_list.append(text)


        initial_text = "a"
        if textbox == True:
            axbox = fig.add_subplot(gs[4])
            text_box = TextBox(axbox, 'GCP Name', initial=initial_text)
            text_box.on_submit(on_submit)


        

        def onclick(event):
            
            if event.dblclick:
                print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                    (event.button, event.x, event.y, event.xdata, event.ydata))
                
                point = []                  #create and fill a list for x,y data for a point
                point.append(event.xdata)
                point.append(event.ydata)

                pair.append(point) #add this point to the list of control points

                plt.plot(event.xdata, event.ydata, 'o')
                fig.canvas.draw()
                #print("Points:", pair)

            if event.button == 3:
                plt.close(fig)
                print('Figure closed')

        def onclick_trajectory(event):

            if event.dblclick == 1:
                print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                    (event.button, event.x, event.y, event.xdata, event.ydata))
                
                point = []                  #create and fill a list for x,y data for a point
                point.append(event.xdata)
                point.append(event.ydata)
                print("Point clicked on:", point)

                pair.append(point) #add this point to the list of control points
                print("Point added to pair")

                plt.plot(event.xdata, event.ydata, "o")
                if len(pair) >=2:
                    x = []
                    y = []
                    for i in range(len(pair)):
                        x.append(pair[i][0])
                        y.append(pair[i][1])

                    plt.plot(x, y, ls = "-")
                fig.canvas.draw()

            if event.button == 3:
                print(text_list)
                plt.close(fig)
                print('Figure closed')

        def onclick_correction_check(event):
            if event.dblclick == 1:
                print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                    (event.button, event.x, event.y, event.xdata, event.ydata))
                
                point = []                  #create and fill a list for x,y data for a point
                point.append(event.xdata)
                point.append(event.ydata)
                print("Point clicked on:", point)

                pair.append(point) #add this point to the list of control points
                print("Point added to pair")

                plt.plot(event.xdata, event.ydata, "o")
                if len(pair) >=2: #if there are two or more points, then plot a line between them
                    x = []
                    y = []
                    for i in range(len(pair)):
                        x.append(pair[i][0])
                        y.append(pair[i][1])

                    plt.plot(x, y, ls = "-")
                
                fig.canvas.draw()

            if event.button == 3:

                plt.close(fig)
                print('Figure closed')


        def close_piece(event):
            nonlocal message
            if event.key == "x":
                message = 1
                print("The x key was pressed and message = ", message)
                plt.close(fig)
                print("Next!")

        #def close_program(event):
            nonlocal done
            if event.key == "d":
                done = 1
                print('The d key was pressed and the program was closed')
                plt.close(fig)

        #cid = fig.canvas.mpl_connect("key_press_event", close_program) #connects the key event with the cations in onclick
            
        if basic_points == True:
            cid = fig.canvas.mpl_connect('button_press_event', onclick) #connects the button event with the actions in onclick

        if trajectory == True:
           cid = fig.canvas.mpl_connect('button_press_event', onclick_trajectory) #connects the button event with the actions in onclick

        if close_piece_option == True:
            cid = fig.canvas.mpl_connect("key_press_event", close_piece) # connects key event with actions in close_piece

        if correction_check == True:
           cid = fig.canvas.mpl_connect('button_press_event', onclick_correction_check) #connects the button event with the actions in onclick

        fig.canvas.toolbar.zoom()
        plt.show() #creates the plot
        
        print('message= ', message)
        print('done?', done)
        

        return pair, text_list[0], message, done
    

            

class ImagePreparation:
    def __init__(self):

            self.lens_params = None      #create a class variable for lend parameters that is empty unless specified when initializing the class
            self.control_points_flume = None
            self.control_points_image = None
            self.images_to_correct_dn =None

            print("Initialized")
    
    def brightenImages(self, images_dn, save_as_dn, alpha = 1.5, beta = 15):
        import cv2
        import glob
        import os
        
        images = glob.glob(images_dn + '/*.jpg')
        for fname in images:
            img = cv2.imread(fname)

            # call convertScaleAbs function
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
           
            # Save and display the transformed image
            fn = os.path.basename(fname).split('.')[0]
            filename = save_as_dn + '\\' + fn + "_brightened.jpg"
            #cv2.imshow("transformed", adjusted)
            #cv2.waitKey(0)
            cv2.imwrite(filename, adjusted)
            cv2.destroyAllWindows()

    def brightenVideo(self, start_time, clip_duration, step = 1/24, alpha = 1.5, beta = 15):
        import cv2
        import os

        input_video = Managers.loadFn("Choose a video file")
        output_dn = Managers.loadDn("Choose a directory to store brightened video in")

        fn = os.path.basename(input_video).split('.')[0]
        output_path = output_dn + '\\' + fn + "_brightened.mp4"

        start_time = start_time
        clip_duration = clip_duration
        step = step
        count = start_time 
        success = True

        cap = cv2.VideoCapture(input_video)

        # Get the video's frames per second and frame size
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while success and count <= start_time + clip_duration:
            cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 

            if success == True:
                ret, frame = cap.read()

                # Increase brightness using cv2.addWeighted
                brightened_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

                # Write the brightened frame to the output video
                out.write(brightened_frame)

                count = count + step
                print(count)

            # Break the loop if no more frames are available
            else:
                break

        # Release video capture and writer objects
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()


    def findImageControlPoints(self, save_image_control_points=True, ):
        import numpy as np

        control_points, message = Managers.pointSelection(self.image_for_control_points, basic_points = True)
        print(control_points)

        self.control_points_image = control_points

        if save_image_control_points == True:
            image_control_points_save_as =  Managers.loadDn('Choose directory to place image control points file into') + "//" + "image_control_points.csv"

        np.savetxt(image_control_points_save_as, control_points, delimiter = ",")
                     

    def loadFlumeControlPoints(self, control_points_fn):
        import csv 
        import numpy as np

        with open(control_points_fn, 'r') as cp_csv:
            csv_reader = csv.reader(cp_csv) 
            # convert string to list 
            control_points = list(csv_reader) 
            cp = np.array(control_points)
            cp_float = cp.astype(float)

        self.control_points_flume= cp_float

        print("Flume control Points:", cp_float)

    def loadImageControlPoints(self, control_points_fn):
        import csv 
        import numpy as np

        with open(control_points_fn, 'r') as cp_csv:
            csv_reader = csv.reader(cp_csv) 
            # convert string to list 
            control_points = list(csv_reader) 
            cp = np.array(control_points)
            cp_float = cp.astype(float)

        self.control_points_image= cp_float

        print("Image Control Points:", cp_float)


    def extractImages(self, input_vid, image_sequence_output_dn, start_time, clip_duration, step):
        import os
        import cv2 
        
        path_in = input_vid
        path_out = image_sequence_output_dn
        start_time = start_time
        clip_duration = clip_duration
        step = step
        count = start_time

        vidcap = cv2.VideoCapture(path_in)
        success,image = vidcap.read()
        success = True
        seq_frame_number = 0
  
        while success and count <= start_time + clip_duration:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
            success,image = vidcap.read()
            print ('Read a new frame: ', success)
            fn = os.path.basename(path_in).split('.')[0]
            if success == True:
                cv2.imwrite(path_out + "\\" + fn + "_frame%d.jpg" % seq_frame_number, image)     # save frame as JPEG file
                count = count + step
                print(count)
                seq_frame_number += 1
                print(seq_frame_number)
                print("Frame written")

            else:
                continue
            

    def calibrateLens(self, checkerboard, calibration_images_dn, 
                      save_calibration_params = False, show_checkerboard_id = False):
        import cv2
        import numpy as np
        import glob
        import csv

        CHECKERBOARD = checkerboard
        directory = calibration_images_dn
    
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.


            # Extracting path of individual image stored in a given directory
        images = glob.glob(directory + '/*.png')
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
                if show_checkerboard_id == True:
                    cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(1000)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
            rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

        params = {"DIM": _img_shape[::-1], "K": K.tolist(), "D": D.tolist()}
            
        if save_calibration_params == True:
            cal_params_save_as =  Managers.loadDn('Choose directory to place calibration parameters file into') + "//" + "calibration_parameters.csv"

            with open(cal_params_save_as, "w") as csvfile:
                w = csv.DictWriter(csvfile, params.keys())
                w.writeheader()
                w.writerow(params)
                     
        self.lens_params = params
        print(self.lens_params)

    def loadCalibrationParams(self, cal_params_fn):
        import csv

        with open(cal_params_fn, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            params = [row for row in reader]
        self.lens_params = params


    def defish(self, imagestocorrect = None, imagetodefish = None, corrected_images_DN = None, show_defished = False): #velocitiesd to be undistort
        import cv2
        import numpy as np
        import glob
        import os
        import ast
        import matplotlib.pyplot as plt

        K = np.array(ast.literal_eval(self.lens_params[0]['K']))
        D = np.array(ast.literal_eval(self.lens_params[0]['D']))
        DIM = tuple(ast.literal_eval(self.lens_params[0]['DIM']))

        if imagestocorrect != None:
            images = glob.glob(imagestocorrect + '/*.jpg')
            for fname in images:
                img = cv2.imread(fname)

                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
                undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     #Remove the fisheye effect from each image

                #find just the file name (no directory or file extension)
                fn = os.path.basename(fname).split('.')[0]
            
                filename = corrected_images_DN + '\\' + fn + "_defished.jpg"

                cv2.imwrite(filename, undistorted_img)
                cv2.destroyAllWindows()

                if show_defished == True:
                    cv2.imshow("undistorted", undistorted_img)
                    cv2.waitKey(500)

            self.images_to_correct_dn = corrected_images_DN

        if imagetodefish != None:
            img = cv2.imread(imagetodefish)

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)     #Remove the fisheye effect from each image

            self.image_for_control_points = undistorted_img

            cv2.imshow("undistorted", self. image_for_control_points)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            


    def transform(self, save_as_dn):
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import glob

        # Specify input and output coordinates that is velocitiesd
        # to calculate the transformation matrix
        input_pts = np.float32(self.control_points_image)
        output_pts = np.float32(self.control_points_flume)

        print("Input points", input_pts)
        print("Output points", output_pts)

        images = glob.glob(self.images_to_correct_dn + '/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            # Compute the perspective transform M
            M = cv2.getPerspectiveTransform(input_pts,output_pts)
                    
            # Apply the perspective transformation to the image
            out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
                    
            # Save and display the transformed image
            fn = os.path.basename(fname).split('.')[0]
            filename = save_as_dn + '\\' + fn + "_transformed.jpg"
            cv2.imshow("transformed", out)
            cv2.waitKey(0)
            cv2.imwrite(filename, out)
            cv2.destroyAllWindows()

    #I need a class to estimate the error image correction

class ImageAnalysis:
    def __init__(self):
        print("Initialized")

    
    def correction_analysis(self, corrected_image_fn, iterations):

        import cv2
        import math
        import statistics as stats

        pairs = [] #list of pairs, will be as long as the number of distances that you want to check
            
        image = cv2.imread(corrected_image_fn)

        for i in range(iterations):
            pair = []
            
            pair, message = Managers.pointSelection(image, correction_check=True)

            print("pair", pair)

            pairs.append(pair)

        print ("Pairs of points:", pairs)
        
        #now lets do something with these point pairs: lets find the distance between them (in pixels)
        distances = []
        
        for i in range(len(pairs)):
            print(pairs[i])
            point1 = pairs[i][0]
            point2 = pairs[i][1]
            #print("Point 1:", point1)
            #print("point 2", point2)

            distance = math.dist(point1, point2)

            distances.append(distance)

        print(distances)
        mean_distance = stats.mean(distances)
        sd_of_distances = stats.stdev(distances)
            
        print("Mean Distance Between dowels:", mean_distance)
        print("Standard Deviation of Distance Measurements", sd_of_distances)





    #I need a function for creating particle path data
    def trackWood(self, images_dn, iterations, output_dn, pixel_size, starting_frame):
        import matplotlib.pyplot as plt
        import glob
        import cv2
        import csv
        import numpy as np
        import os

        trajectory_data = {}

        for i in range(iterations):
            pairs = []
            message = 0
            subset_of_images = []
            images = glob.glob(images_dn + '/*.jpg')
            # Define a custom sorting key function
            def get_numeric_part(filename):
                return int(''.join(filter(str.isdigit, filename)))

            # Sort the files based on the numeric part of the filename
            sorted_images = sorted(images, key=get_numeric_part)

            # Select a subset of the sorted images based on indices
            subset_of_images = sorted_images[starting_frame:]
            #print("Subset of Images:", subset_of_images)
            #print("Length of Images:", len(subset_of_images))


            for fname in subset_of_images:


                img = cv2.imread(fname)
                pair, message = Managers.pointSelection(img, trajectory= True, close_piece_option=True)

                if message == 1:
                    print("Message = 1 and loop will break")
                    break 

                print("Pair from piece ", i, " in image ",fname, ":", pair)

                if pair != []:
                    pairs.append(pair)
                
                print("Pair added to pairs:", pair)

            trajectory_data[i] = np.array(pairs)
            print("trajectory data:", trajectory_data)
            
        print("Trajecotry Data in Image Coordinates", trajectory_data)

        real_world_trajectories = {}
        for i in range(len(trajectory_data)):
            real_world_coords = trajectory_data[i]*pixel_size
            real_world_trajectories[str(i)] = real_world_coords
            
        print("Trajectory data in Real world coordinates:", real_world_trajectories)

        csv_directory = output_dn        

        # Save each NumPy array to a separate CSV file
        for key, array in real_world_trajectories.items():
            csv_file_path = os.path.join(csv_directory, f'{key}')
            csv_file_path_final = csv_file_path + "_" + str(starting_frame)
            print(csv_file_path_final)
            
            
            # Save 3D array to CSV file
            np.save(csv_file_path_final, array)

        print(f'Data has been written to {output_dn}')


    def findVelocities(self, trajectories_dn, step, output_dn):
        import csv
        import numpy as np
        import ast
        import math
        import numpy as np
        import os


        # Initialize an empty dictionary to store the NumPy arrays
        trajectories = {}

        # Loop through each file in the directory
        for file_name in os.listdir(trajectories_dn):
            if file_name.endswith(".npy"):
                file_path = os.path.join(trajectories_dn, file_name)
                
                # Load data from the file
                array_data = np.load(file_path)
                
                # Extract array name from the file name
                array_name = os.path.splitext(file_name)[0]
                
                # Store the NumPy array in the dictionary
                trajectories[array_name] = array_data

        print('Data has been loaded:')
        print(trajectories)

        #now its time to start calculating velocities between points
        velocities = {}

        for key in trajectories:              #iterating over each piece
            piece = trajectories[key]
            print("Piece:", key)
            print("Data", piece)


            us = []

            for j in range(len(piece) - 1):  #you will iterate 1 less than the number of points that you have 
                print(j)
                #print ("time", j, "to time", j+1)
                #print("Piece", piece)
                #print("piece[j]", piece[j])
                x_data = []
                y_data = []
                x_position = None
                y_position = None

                #find midpoint of each piece
                point1_x = (piece[j][0][0] + piece[j][1][0])/2
                point1_y = (piece[j][0][1] + piece[j][1][1])/2
                point_1 = [point1_x, point1_y]

                point2_x = (piece[j+1][0][0] + piece[j+1][1][0])/2
                point2_y = (piece[j+1][0][1] + piece[j+1][1][1])/2
                point_2 = [point2_x, point2_y]

                print( "point 1", point_1)
                print("point 2", point_2)

                x_data.append(point1_x)
                x_data.append(point2_x)

                y_data.append(point1_y)
                y_data.append(point2_y)

                #create a np array of y positions
                x_position = np.array(x_data)
                y_position = np.array(y_data)
                print("x", x_position)
                print("y", y_position)

                # Calculate velocity
                x_velocity = np.gradient(x_position, step)
                y_velocity = np.gradient(y_position, step)
                velocity = np.sqrt(x_velocity**2 + y_velocity**2)

                print('Velocity:', velocity)
                us.append(velocity[0])
                print("us", us)

            us = np.array(us)
            velocities[key] = us

        print(velocities)

        # Specify the CSV file path
        csv_file = output_dn + '\\velocities.csv'

        # Find the maximum length among all arrays
        max_length = max(len(arr) for arr in velocities.values())

        # Create a list of lists to store the data
        data_list = []

        # Iterate over the arrays and pad them with NaN to the maximum length
        for key, array in velocities.items():
            padded_array = np.pad(array, (0, max_length - len(array)), mode='constant', constant_values=np.nan)
            data_list.append([key] + padded_array.tolist())

        # Write the data to the CSV file
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the data
            writer.writerows(data_list)