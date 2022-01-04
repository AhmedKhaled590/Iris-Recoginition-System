import numpy as np
import cv2 as cv
import matplotlib as mpl
def log_gabor (img,min_wave_len,sigma):
    #get width and height of image
    rows,col=img.shape 
    # print('rows,col: ',rows,col)
    #intiate gabor filter as an array with the # of columns 
    log_gabor=np.zeros(col)
    # print('log gabor: ',log_gabor)
    filter_bank = np.zeros([rows, col], dtype=complex)
    # print('filter_bank: ',filter_bank)
    radius=np.arange(col/2 + 1) / (col/2) / 2
    # print('radius: ',radius)
    radius[0]=0
    # print('radius: ',radius)
    wave_len=min_wave_len
    fo=1/wave_len
    log_gabor[0 : int(col/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigma)**2))
    # print('log_gabor: ',log_gabor)
    log_gabor[0]=0

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
            # print('img inside for',img.shape)
            signal = img[r, 0:col]
            # print('signal: ',signal)
            # log_gabor is a pass filter and we want to modulate
            #  the signal with it so we convert signal to frequency domain
            #  as convolution is in spatial domain is multiplication in 
            #  the frequency domain
            fourier_transform_of_signal = np.fft.fft(signal)
            # print('fft signal',fourier_transform_of_signal)
            #we use inverse fourier transform to get the 
            # filtered signal in spatial domain
            filter_bank[r , :] = np.fft.ifft(fourier_transform_of_signal * log_gabor)
            # print('filter bank: ',filter_bank)
    return filter_bank



def normalization_to_template (arr_in_polar,noise_arr,min_wave_len,sigma):
        # print("inside template",arr_in_polar.shape[0])
        arr_in_polar = cv.cvtColor(arr_in_polar, cv.COLOR_BGR2GRAY)
        filter_bank=log_gabor(arr_in_polar,min_wave_len,sigma)
        len=filter_bank.shape[1]
        temp = np.zeros([arr_in_polar.shape[0], 2 * len])
        # h = np.arange(arr_in_polar.shape[0])
        mask = np.zeros(temp.shape)

        eleFilt = filter_bank[:, :]
        H1 = np.real(eleFilt) > 0
        H2 = np.imag(eleFilt) > 0
        H3 = np.abs(eleFilt) < 0.0001
        for i in range(len):
                ja = 2 * i
                temp[:, ja] = H1[:, i]
                temp[:, ja + 1] = H2[:, i]
                mask[:, ja] = noise_arr[:, i] | H3[:, i]
                mask[:, ja + 1] = noise_arr[:, i] | H3[:, i]
        return temp,mask





        

 




        