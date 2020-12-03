import numpy as np
from PIL import Image, ImageOps
from skimage import filters
import matplotlib.pyplot as plt
import os
import re
import csv
from datetime import datetime, date, time

### SATURATION CALCULATIONS ###

# go through all webcam images
raw_image_dir = 'Cleveland_Webcam_Images'
output_image_path = 'Output/'
ref_image_path = 'References/'

reference_image = Image.open(ref_image_path + 'Cleveland_webcam_reference.png')
reference_image_array = np.asarray(reference_image)

# Create and append the cloud detection images
fields=["yyyy-mm-dd", "time(UTC)", "Night/Fog(True/False)", "high_left_clouds(True/False)", "high_left_saturation(%)" ,"high_right_clouds(True/False)", "high_right_saturation(%)","low_left_clouds(True/False)", "low_left_saturation(%)", "low_right_clouds(True/False)", "low_right_saturation(%)", "Clear(True/False)"]
with open(output_image_path+'CLCO_Cloud_Detections.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

for image_filename in os.listdir(raw_image_dir):
    tested_image = Image.open(raw_image_dir + '/' + image_filename)
    image_name = image_filename.split('.')[0]
    
    # regular expression to get date and time of webcam image
    p = re.compile(r'cleveland_clco-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})')
    match = p.findall(image_name)
    if not match:
        p = re.compile(r'cleveland_clco-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})')
        match = p.findall(image_name)
    match = match[0]
 

    # Convert image to greyscale then apply sobel edge detection
    grayscale_image = ImageOps.grayscale(tested_image)
    edge_sobel = filters.sobel(grayscale_image)

    #This is zeroing out everything that isn't the pixels that we care about
    with open(ref_image_path + 'reference_arrays.npy', 'rb') as f:
        reference_2D = np.load(f)
        X_reference = np.load(f)
        Y_reference = np.load(f)
    webcam_reduction = reference_2D * edge_sobel
    # this is a spot I know is white
    # print(webcam_reduction[233][298])

    # Determine saturation of each pixel in the array from the reference file
    # use x and y ref to make array of saturations we care about
    absolute_saturation_values = []
    relative_saturation_values = []
    saturation_cutoff = 0.10 # 0.10 removes most noise
    left_abs_dict = {}
    right_abs_dict = {}
    left_rel_dict = {}
    right_rel_dict = {}

    num_interested_pixels = len(X_reference)
    for y_position, pixel_yvalue in enumerate(Y_reference):
        pixel_xvalue = X_reference[y_position]
        absolute_saturation = webcam_reduction[pixel_yvalue][pixel_xvalue]

        if absolute_saturation < saturation_cutoff:
            absolute_saturation_values.append(None)
            relative_saturation_values.append(None)
        else:
            absolute_saturation_values.append(absolute_saturation)
            # this is using a reference webcam image as a comparison, instead of just black/white
            ref_image_saturation = reference_image_array[pixel_yvalue][pixel_xvalue][0]
            relative_saturation = absolute_saturation / (ref_image_saturation/255.0)
            relative_saturation_values.append(relative_saturation)
        
            # splitting between left and right halves of volcano
            if pixel_xvalue < 310:
                if pixel_yvalue in left_abs_dict:
                    abs_val = left_abs_dict[pixel_yvalue]
                    abs_val.append(absolute_saturation)
                    left_abs_dict[pixel_yvalue] = abs_val
                    rel_val = left_rel_dict[pixel_yvalue]
                    rel_val.append(relative_saturation)
                    left_rel_dict[pixel_yvalue] = rel_val
                else:
                    left_abs_dict[pixel_yvalue] = [absolute_saturation]
                    left_rel_dict[pixel_yvalue] = [relative_saturation]
            else:
                if pixel_yvalue in right_abs_dict:
                    abs_val = right_abs_dict[pixel_yvalue]
                    abs_val.append(absolute_saturation)
                    right_abs_dict[pixel_yvalue] = abs_val
                    rel_val = right_rel_dict[pixel_yvalue]
                    rel_val.append(relative_saturation)
                    right_rel_dict[pixel_yvalue] = rel_val
                else:
                    right_abs_dict[pixel_yvalue] = [absolute_saturation]
                    right_rel_dict[pixel_yvalue] = [relative_saturation]

    # averaging points on left and right sides of volcano
    left_ys = []
    left_abs_avgs = []
    left_rel_avgs = []
    for y_val in left_abs_dict:
        left_ys.append(y_val)
        left_abs_avgs.append(np.mean(left_abs_dict[y_val]))
        left_rel_avgs.append(np.mean(left_rel_dict[y_val]))

    right_ys = []
    right_abs_avgs = []
    right_rel_avgs = []
    for y_val in right_abs_dict:
        right_ys.append(y_val)
        right_abs_avgs.append(np.mean(right_abs_dict[y_val]))
        right_rel_avgs.append(np.mean(right_rel_dict[y_val]))
        
	
    #### METRICS FOR QUALITY OF IMAGE ###
    
    # define everything initially
    high_left_clouds = None
    low_left_clouds = None
    high_right_clouds = None 
    low_right_clouds = None
    TL_percent_saturated = None
    BL_percent_saturated = None
    TR_percent_saturated = None
    BR_percent_saturated = None
    
    #Code segment for fog and night detection
    unique_abs_vals = set(left_abs_avgs)
    unique_abs_vals.update(right_abs_avgs)
    fog_night_detection = False

    if len (unique_abs_vals) == 0:
        fog_night_detection = True
    else: # only check for clouds if it's not night/fog
    
        #Finding the halfway point in the array
        midway_y_point = Y_reference[int(len(Y_reference)/2)]
        num_unique_y_values = len(set(Y_reference))
        saturated_pixel_proportion_cutoff = 0.6

        # TOP LEFT QUADRANT
       
        TL_quadrant = []
        for y_position, yvalue in enumerate(left_ys):
            if yvalue >= midway_y_point:
                TL_quadrant.append(left_abs_avgs[y_position])
        
        TL_percent_saturated = len(TL_quadrant) / (num_unique_y_values/2)
        high_left_clouds = True
        if TL_percent_saturated > saturated_pixel_proportion_cutoff:
            high_left_clouds = False
        #print(TL_percent_saturated, high_left_clouds, image_name)
        
    
        #TOP RIGHT QUADRANT
    
        TR_quadrant = []
        for y_position, yvalue in enumerate(right_ys):
            if yvalue >= midway_y_point:
                TR_quadrant.append(right_abs_avgs[y_position])
        
        TR_percent_saturated = len(TR_quadrant) / (num_unique_y_values/2)
        high_right_clouds = True
        if TR_percent_saturated > saturated_pixel_proportion_cutoff:
            high_right_clouds = False
        #print(TL_percent_saturated, high_right_clouds, image_name)
    
    
        #BOTTOM LEFT QUADRANT
        
        BL_quadrant = []
        for y_position, yvalue in enumerate(left_ys):
            if yvalue < midway_y_point:
                BL_quadrant.append(left_abs_avgs[y_position])
        
        BL_percent_saturated = len(BL_quadrant) / (num_unique_y_values/2)
        low_left_clouds = True
        if BL_percent_saturated > saturated_pixel_proportion_cutoff:
            low_left_clouds = False
        #print(TL_percent_saturated, bottom_left_clouds, image_name)
        
                
        #BOTTOM RIGHT QUADRANT
      
        BR_quadrant = []
        for y_position, yvalue in enumerate(right_ys):
            if yvalue < midway_y_point:
                BR_quadrant.append(right_abs_avgs[y_position])
        
        BR_percent_saturated = len(BR_quadrant) / (num_unique_y_values/2)
        low_right_clouds = True
        if BR_percent_saturated > saturated_pixel_proportion_cutoff:
            low_right_clouds = False
        #print(TL_percent_saturated, bottom_right_clouds, image_name)        
        
    
    #Code segment for clear sky
    clear_sky = False
    if fog_night_detection == False and high_left_clouds == False and high_right_clouds == False and low_left_clouds == False and low_right_clouds == False:
        clear_sky = True

    ### PUTTING METRICS INTO A SPREADSHEET ###

    if match:
        fields=[date(int(match[0]), int(match[1]), int(match[2])).isoformat(), time(int(match[3]), int(match[4])).isoformat(), fog_night_detection, high_left_clouds, TL_percent_saturated, high_right_clouds, TR_percent_saturated, low_left_clouds, BL_percent_saturated, low_right_clouds, BR_percent_saturated, clear_sky]
        with open(output_image_path+'CLCO_Cloud_Detections.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        
    ### PLOTTING ###

    # # Graph image by saturation and y coordinate
    # plt.scatter(absolute_saturation_values, Y_reference, s=1, c='Black', marker='s')
    # plt.xlim(0, 1)
    # plt.ylim(340, 233)
    # plt.title(image_name + ' Absolute Saturation')
    # plt.title('Absolute Saturation')
    # filename = output_image_path + image_name + '_absolute_saturation_plot.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()

    # plt.scatter(relative_saturation_values, Y_reference, s=1, c='Black', marker='s')
    # plt.xlim(0, 20)
    # plt.ylim(340, 233)
    # plt.title(image_name + ' Relative Saturation')
    # plt.title('Relative Saturation')
    # filename = output_image_path + image_name + '_relative_saturation_plot.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()

    # # plot averaged saturation
    # plt.scatter(left_abs_avgs, left_ys, s=1, c='Green', marker='s', label='Left Side')
    # plt.xlim(0, 1)
    # plt.ylim(340, 233)
    # plt.scatter(right_abs_avgs, right_ys, s=1, c='Blue', marker='s', label='Right Side')
    # plt.legend(fontsize='medium', markerscale=5)
    # plt.title(image_name + ' Absolute Saturation, Averaged')
    # plt.title('Absolute Saturation, Averaged')
    # filename = output_image_path + image_name + '_absolute_saturation_averaged.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()

    # plt.scatter(left_rel_avgs, left_ys, s=1, c='Green', marker='s', label='Left Side')
    # plt.xlim(0, 20)
    # plt.ylim(340, 233)
    # plt.scatter(right_rel_avgs, right_ys, s=1, c='Blue', marker='s', label='Right Side')
    # plt.legend(fontsize='medium', markerscale=5)
    # plt.title(image_name + ' Relative Saturation, Averaged')
    # plt.title('Relative Saturation, Averaged')
    # filename = output_image_path + image_name + '_relative_saturation_averaged.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()


    # # plot averaged saturation with cutoff
    # plt.scatter(left_abs_avgs, left_ys, s=1, c='Green', marker='s', label='Left Side')
    # plt.xlim(0, 1)
    # plt.ylim(340, 233)
    # plt.scatter(right_abs_avgs, right_ys, s=1, c='Blue', marker='s', label='Right Side')
    # plt.legend(fontsize='medium', markerscale=5)
    # plt.title(image_name + ' Absolute Saturation with 0.1 Cutoff, Averaged')
    # plt.title('Absolute Saturation with 0.1 Cutoff, Averaged')
    # filename = output_image_path + image_name + '_absolute_saturation_averaged_with_cutoff.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()

    # plt.scatter(left_rel_avgs, left_ys, s=1, c='Green', marker='s', label='Left Side')
    # plt.xlim(0, 20)
    # plt.ylim(340, 233)
    # plt.scatter(right_rel_avgs, right_ys, s=1, c='Blue', marker='s', label='Right Side')
    # plt.legend(fontsize='medium', markerscale=5)
    # plt.title(image_name + ' Relative Saturation with 0.1 Cutoff, Averaged')
    # plt.title('Relative Saturation with 0.1 Cutoff, Averaged')
    # filename = output_image_path + image_name + '_relative_saturation_averaged_with_cutoff.jpeg'
    # plt.savefig(filename)
    # # plt.show()
    # plt.close()



    # # plot several images for comparison (3X1 graph)
    # fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(50, 25))
    # ax = plt.subplot('311')
    # ax.set_title('Webcam Image')
    # ax.imshow(tested_image)

    # ax = plt.subplot('312')
    # ax.set_title("Edge Detection")
    # ax.imshow(edge_sobel, cmap=plt.cm.gray)

    # ax = plt.subplot('313')
    # ax.set_title('Scatter Image')
    # scatter_image = Image.open(output_image_path + image_name + '_absolute_saturation_plot.jpeg')
    # ax.imshow(scatter_image)
    # ax.axis('off')

    # plt.tight_layout()    
    # # plt.show()
    # fig.savefig(output_image_path + image_name + '_edgeplot.jpeg')



    # # plot all images for comparison (4X2 graph)
    # fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(15, 25))
    # ax = plt.subplot('421')
    # ax.set_title('Webcam Image')
    # ax.imshow(tested_image)

    # ax = plt.subplot('422')
    # ax.set_title("Edge Detection")
    # ax.imshow(edge_sobel, cmap=plt.cm.gray)

    # ax = plt.subplot('423')
    # image = Image.open(output_image_path + image_name + '_absolute_saturation_plot.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # ax = plt.subplot('424')
    # image = Image.open(output_image_path + image_name + '_relative_saturation_plot.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # ax = plt.subplot('425')
    # image = Image.open(output_image_path + image_name + '_absolute_saturation_averaged.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # ax = plt.subplot('426')
    # image = Image.open(output_image_path + image_name + '_relative_saturation_averaged.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # ax = plt.subplot('427')
    # image = Image.open(output_image_path + image_name + '_absolute_saturation_averaged_with_cutoff.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # ax = plt.subplot('428')
    # image = Image.open(output_image_path + image_name + '_relative_saturation_averaged_with_cutoff.jpeg')
    # ax.imshow(image)
    # ax.axis('off')

    # plt.tight_layout()
    # plt.title('Cleveland ' + timestamp)
    # # plt.show()
    # fig.savefig(output_image_path + image_name + '_comparisons.jpeg')
