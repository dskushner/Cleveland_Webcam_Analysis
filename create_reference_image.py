#This is making the reference image
import numpy as np
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio

from skimage import filters
from skimage.data import camera
import skimage.util
# from skimage.util import compare_images
# #install pillow
# import sys
# !{sys.executable} -m pip install Pillow
# import PIL as pil
from PIL import Image, ImageOps

# Path = "Desktop/Cleveland_Webcam_Images/"
# Copy_to_path = "Desktop/Cleveland_Webcam_Images/"

# # reference_image_filename = "Desktop/Cleveland_Webcam_Images/image34.jpg"
# reference_image_filename = "Cleveland_Webcam_Images/image34.jpg"

# # for image in Path:
# raw_image = Image.open(reference_image_filename)
# image_array = np.asarray(raw_image)
# reference_grayscale_image = ImageOps.grayscale(raw_image)

# # Apply sobel filter
# reference_edge_sobel = filters.sobel(reference_grayscale_image)

# # Subplot for image comparisson
# # fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True,
# #                          figsize=(50, 25))

# # axes[0].imshow(image)
# # axes[0].set_title('Webcam Image')

# # axes[2].imshow(edge_roberts, cmap=plt.cm.gray)
# # axes[2].set_title('Roberts Edge Detection')

# # axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
# # axes[1].set_title('Sobel Edge Detection')
# # fig.savefig("Desktop/Cleveland_Webcam_Images/image34.jpeg")


# #This is saving the reference sobel image (made above) as a 640X480 image
# image_3D = Image.fromarray(np.uint8(cm.gist_earth(reference_edge_sobel)*255))
# grayscale_image = ImageOps.grayscale(image_3D)
# # grayscale_image.save("Desktop/Cleveland_Webcam_Images/Cleveland_Webcam_edge.png")
# grayscale_image.save("References/Cleveland_Webcam_edge.png")


# #This section converts all nonblack pixels to black, and vice versa
# reference_image = plt.imread('Desktop/Cleveland_Webcam_Images/Cleveland_Webcam_reference.png')
reference_image = plt.imread('References/Cleveland_Webcam_reference.png')
# plt.imshow(reference_image)
# plt.show()

image_copy = reference_image.copy()
black_pixels_mask = np.all(reference_image == [0, 0, 0], axis=-1)
non_black_pixels_mask = np.any(reference_image != [0, 0, 0], axis=-1)  
# or non_black_pixels_mask = ~black_pixels_mask
image_copy[black_pixels_mask] = [0, 0, 0]
image_copy[non_black_pixels_mask] = [255, 255, 255]
# plt.imshow(image_copy)
# plt.show()

rescaled_image_copy = (255.0 / image_copy.max() * (image_copy - image_copy.min())).astype(np.uint8)
im_rescaled_image_copy = Image.fromarray(rescaled_image_copy)
# im_rescaled_image_copy.save('Desktop/Cleveland_Webcam_Images/Cleveland_Webcam_reference_mask.jpeg')
im_rescaled_image_copy.save('References/Cleveland_Webcam_reference_mask.jpeg')

# pim = Image.open('image.png').convert('RGB')
im  = np.array(image_copy)

# Define the blue colour we want to find - PIL uses RGB ordering
white = [255,255,255]

# Get X and Y coordinates of all black pixels
Y_reference, X_reference = np.where(np.all(im==white,axis=2))

# zipped = np.column_stack((X,Y))

newarray = im/255

# print(im)

reference_2D = newarray[:,:,0]
# print(np.shape(reference_2D))
# print(reference_2D[233][298])

#verification that the array is good by recreating a plot with the array
# plt.scatter( X, Y, s=1, c='Black', marker="s")
# axes = plt.gca()
# axes.set_xlim([0,640])
# axes.set_ylim([480,0])
# plt.show()

#save array to CSV
# np.savetxt("Desktop/Cleveland_Webcam_Images/Celeveland_pixelreferenceY.csv", Y, delimiter=",")
with open('References/reference_arrays.npy', 'wb') as f:
    np.save(f, reference_2D)
    np.save(f, X_reference)
    np.save(f, Y_reference)
