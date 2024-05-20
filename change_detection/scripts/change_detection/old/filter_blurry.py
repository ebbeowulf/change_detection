import os
import argparse
import cv2
import numpy as np
import pdb

def calculate_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return value

def calculate_blur_fft(img, size=60):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    h=image.shape[0]
    w=image.shape[1]
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

# def is_blur(image) :
#    """
#    This function convolves a grayscale image with
#    laplacian kernel and calculates its variance.
#    """

#    thresold = #Some value you need to decide

#    #Laplacian kernel
#    laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
#    laplacian_kernel = tf.expand_dims(laplacian_kernel, -1)
#    laplacian_kernel = tf.expand_dims(laplacian_kernel, -1)
#    laplacian_kernel = tf.cast(laplacian_kernel, tf.float32)

#    #Convolving image with laplacian kernel
#    new_img = tf.nn.conv2d(image, laplacian_kernel, strides=[1, 1, 1, 1], 
#                           padding="SAME")

#    #Calculating variance
#    img_var = tf.math.reduce_variance(new_img)

#    if img_var < thresold :
#       return True
#    else :
#       return False
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir',type=str,help='location of images in file system')
    parser.add_argument('--threshold',type=float,default=20.0,help="default blur threshold")
    args = parser.parse_args()

    all_images=os.listdir(args.image_dir)

    for im in all_images:
        fName=args.image_dir + "/" + im
        try:
            image_color = cv2.imread(fName)
            blur=calculate_blur(image_color)
            # print("%s - %f"%(fName,blur))
            # if blur<args.threshold:
            #     cv2.imshow("blurry", image_color)
            #     cv2.waitKey(0)
            if blur<args.threshold:
                print(fName)
        except Exception as e:
            print("Error opening file - " + fName)
            continue