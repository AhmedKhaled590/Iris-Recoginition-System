import skimage.io as io
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy as np
from scipy.signal import convolve2d


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def apply_filter_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    
    # we supply the img shape here to make both the filter and img have the same shape to be able to multiply
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
    filtered_img = fftpack.ifft2(filtered_img_in_freq)
     
    
    show_images([img,
                fftpack.fftshift(np.log(np.abs(img_in_freq)+1)), # log for better intensity scale, 
                                                                 # shift to make zero freq at center
                fftpack.fftshift(np.log(np.abs(filter_in_freq)+1)),
                fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1)),
                np.abs(filtered_img)
             ], ['Image', 'Image in Freq. Domain', 'Filter in Freq. Domain', 'Filtered Image in Freq. Domain', 'Filtered Image'])
    return  filtered_img

def sobel_filters(img):
    hy = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])
                        
    hx= np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ]) 
    f=np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ])

    img_x = convolve2d( img, hx)
    img_y = convolve2d(img, hy)
    gradient_magnitude = np.sqrt(np.square(img_x) + np.square(img_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    theta = np.arctan2(img_y, img_x)
    return (gradient_magnitude, theta)
