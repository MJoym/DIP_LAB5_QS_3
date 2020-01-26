import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import entropy as scipy_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure.entropy import shannon_entropy
import cv2 as cv
from array import array

######################################################
################### LAB 5 ############################
######################################################

################## FUNCTIONS ######################


# def myEntropy(img):
#
#     p = hist(img, bins='knuth', alpha=0.2, density=True)
#     # p = p(p > 0)
#     ent = - (np.sum(p[:]*np.log2(p[:])))
#     plt.figure('My Entropy Function Calculation')
#     plt.plot(p)
#     plt.title('Entropy: ', ent)

# Not specifically for a binary image:
def rle_compress(imgList):
    encodedImage = []

    for row in imgList:
        encodedRow = []
        count = 0
        prev = row[0]
        for pixel in row:
            if pixel == prev:
                count += 1
            elif count > 1:
                encodedRow.append([count, prev])
                prev = pixel
                count = 1

        if count > 1:
            encodedRow.append([count, prev])

        encodedImage.append(encodedRow)

    return encodedImage


# For a BINARY image:
def bin_rle_compress(img, flag):
    binaryEncodedImage = []
    count = 0
    prev = img[0]

    if flag == 0:  # This means the first pixel is white
        first_pixel = 0
        binaryEncodedImage.append(first_pixel)

    for pixel in img:
        if pixel == prev:
            count += 1
        elif count > 1:
            binaryEncodedImage.append(count)
            prev = pixel
            count = 1

    if count > 1:
        binaryEncodedImage.append(count)

    return binaryEncodedImage


# Function that retrieves a binary image from a compressed array:
def decoder_rle(ls_img, width, height):
    lst = []

    current = ls_img[0]
    end = len(ls_img)
    enter = True

    if current == 0:  # First pixels are white
        flag = 0
    else:
        flag = 1

    for ls_index in range(0, end-1):
        if flag == 1:  # Black pixels
            lst.append(np.zeros(ls_img[ls_index]))
            enter = False
            flag = 0

        if flag == 0:  # White pixels
            if enter:
                lst.append(np.ones(ls_img[ls_index]) * 255)
                flag = 1

            else:
                enter = True

    count = 0
    for item in lst:
        for el in item:
            arr = array("i", item[el])

    arr2mat = np.asmatrix(arr)
    np.reshape(arr2mat, (width, height))

    return arr2mat


####################################
# 3.1.1 - Lines Image and Entropy
####################################
# N = 200 in example
N = 200
# Create an NxN matrix
I = np.zeros((N, N))

# Create the binary image:
i = 20
I[i:i+160, i:i+160] = 255
I[i+20:i+140, i+20:i+140] = 0
I[i+40:i+120, i+40:i+120] = 255
I[i+60:i+100, i+60:i+100] = 0


# Create second example image:
B = np.zeros((50, 50))
k = 14
B[k:k+22, k:k+22] = 255


# myEntropy(img=I)
# print(shannon_entropy(I))

#####################################
######### RLE COMPRESSION ###########
#####################################

# Set image matrix to be an array (one line)
flat_imB = np.array(B).flatten()
flat_imI = np.array(I).flatten()

# Set image matrix to be a list of lists->each representing a row:
img1 = I.tolist()

# Function that counts how many pixels are repeated and returns a list of N lists (200 in this example)
# each sub-list is a row of the image I and each one contains the (#repetitions, pixel value):
rle_list= rle_compress(img1)

# For the small second binary image NxN = 50x50: if flag = 0, then first pixel is white
# so the output will have a '0' at the beginning meaning the image starts with white pixels:
bin_rle_B = bin_rle_compress(flat_imB, flag=1)
restored_B = decoder_rle(bin_rle_B, 50, 50)
# for the next quwstion in the Lab: compress the first image (I):
bin_rle_I = bin_rle_compress(flat_imI, flag=1)
# restored_I = decoder_rle(bin_rle_I, N, N)



# For the tire.tif compression:
tire = cv.imread('/home/joy/MATLAB/R2019b/toolbox/images/imdata/tire.tif')
ret, bin_tire = cv.threshold(tire, 127, 255, cv.THRESH_BINARY)
flat_tire = np.array(bin_tire).flatten()
bin_tire_compress = bin_rle_compress(flat_tire, flag=1)

print('The Compressed Image using RLE Compression:')
print(rle_list)
print('For a binary image with the first pixel known as black:')
print(bin_rle_B)
print('The compressed signal for tire.tif:')
print(bin_tire_compress)

# Show the binary images created and their histograms:
plt.figure('image #1')
plt.subplot(1,2,1)
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.title('Image #1')
plt.subplot(1,2,2)
plt.hist(I.ravel(), 256, [0, 256])
plt.title('Histogram #1')
plt.figure('second image')
plt.subplot(1,2,1)
plt.imshow(B, cmap='gray')
plt.axis('off')
plt.title('Image #2')
plt.subplot(1,2,2)
plt.hist(B.ravel(), 256, [0, 256])
plt.title('Histogram #2')
plt.figure('Restored I')
plt.imshow(restored_B, cmap='gray')
plt.axis('off')
plt.title('Restored I')
plt.show()
cv.imshow("Binary Tire", bin_tire)
cv.waitKey(0)
cv.destroyAllWindows()

