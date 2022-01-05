import cv2
import numpy as np
# to implement the log-Gabor filter in the image
def filter (img,min_wave_len,sigma):
#convert RGB to gray scale
        
    #get width and height of image
    rows,col=img.shape 
    #intiate gabor filter as an array with the # of columns 
    filter=np.zeros(col)
    #array of complex convolution
    filter_bank = np.zeros([rows, col], dtype=complex)
    
    #frequencies
    radius=np.arange(col/2 + 1) / (col/2) / 2
    radius[0]=0
    #intialize wave length
    wave_len=min_wave_len
    #center frequency of the filter
    fo=1/wave_len
    #at the end of the day the filter is just a gaussian function
    filter[0 : int(col/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigma)**2))
    filter[0]=0

    # Assume this the image 
    # *****************
    # *****************
    # *****************
    # *****************
    # *****************
    # *****************

    # and the filter is #################
    # so what we do is just move the filter row by row

    for r in range(rows):
            signal = img[r, 0:col]
            # filter is a pass filter and we want to modulate
            #  the signal with it so we convert signal to frequency domain
            #  as convolution is in spatial domain is multiplication in 
            #  the frequency domain
            fourier_transform_of_signal = np.fft.fft(signal)
            #we use inverse fourier transform to get the 
            # filtered signal in spatial domain
            filter_bank[r , :] = np.fft.ifft(fourier_transform_of_signal * filter)
    return filter_bank



def normalization_to_template (arr_in_polar,noise,min_wave_len,sigma):
        # we implement the filter on the important data (normalized one)(we extract the
        #  noise from the segmented 
        # image) to extract the features of the image
        #convert RGB to gray scale of arr_in_polar
        arr_in_polar=cv2.cvtColor(arr_in_polar,cv2.COLOR_RGB2GRAY)
        noise=cv2.cvtColor(noise,cv2.COLOR_RGB2GRAY)
        filter_bank=filter(arr_in_polar,min_wave_len,sigma)
        len=filter_bank.shape[1]
        temp = np.zeros([arr_in_polar.shape[0], 2 * len])
        mask = np.zeros(temp.shape)
        eleFilt = filter_bank[:, :]
        #the real part of the filter bank
        H1 = np.real(eleFilt) > 0
        #the imaginary part of the filter bank
        H2 = np.imag(eleFilt) > 0
        # If amplitude is close to zero then phase data is not useful,
	# so mark off in the noise mask
        H3 = np.abs(eleFilt) < 0.0001
        for i in range(len):
                ja = 2 * i
                #temp of index ja is the real part of the filter bank
                temp[:, ja] = H1[:, i]
                temp[:, ja + 1] = H2[:, i]
                mask[:, ja] = noise[:, i] | H3[:, i]
                mask[:, ja + 1] = noise[:, i] | H3[:, i]
        return temp, mask

