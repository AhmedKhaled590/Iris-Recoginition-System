from scipy.ndimage.filters import convolve
import numpy as np
from skimage.color.colorconv import gray2rgb
import skimage.io as io
from sobel import sobel_filters
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from collections import defaultdict
from skimage.color import rgb2gray
from scipy import signal
from matplotlib import pyplot as plt

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.keep=set()
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                            self.keep.add((i,j))
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return (img, self.keep)
    
    def detect(self):
        for i, img in enumerate(self.imgs):    
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final,self.keep = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return (self.imgs_final, self.keep)

##-----------------------------------------------------------------------------
##  Function
##-----------------------------------------------------------------------------
def searchInnerBound(img):
    """
    Description:
        Search for the inner boundary of the iris.
    Input:
        img		- The input iris image.
    Output:
        inner_y	- y-coordinate of the inner circle centre.
        inner_x	- x-coordinate of the inner circle centre.
        inner_r	- Radius of the inner circle.
    """

    # Integro-Differential operator coarse (jump-level precision) to find first impression
    Y = img.shape[0]
    X = img.shape[1]
    sect = X/4 		# Width of the external margin for which search is excluded
    minrad = 10
    maxrad = sect*0.8
    jump = 4 		# Precision of the coarse search, in pixels

    # Hough Space (y,x,r)
    sz = np.array([np.floor((Y)/jump),
                    np.floor((X)/jump),
                    np.floor((maxrad-minrad)/jump)]).astype(int)
    # Resolution of the circular integration
    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision) # angels from 0 to 2Pi
    # c 
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    
    y = y*jump    
    x = x*jump   
    r = minrad + r*jump 
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]
    # Blur
    sm = 3 		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y =  y*jump
    inner_x =  x*jump
    inner_r = minrad + (r-1)*jump


    # Integro-Differential operator fine (pixel-level precision)
    integrationprecision = 0.1 		# Resolution of the circular integration
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(inner_x),
                          np.arange(inner_y),
                          np.arange(inner_r))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Bluring
    sm = 2		# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)
    
    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r


#------------------------------------------------------------------------------
def searchOuterBound(img, inner_y, inner_x, inner_r):
    """
    Description:
        Search for the outer boundary of the iris.
    Input:
        img		- The input iris image.
        inner_y	- The y-coordinate of the inner circle centre.
        inner_x	- The x-coordinate of the inner circle centre.
        inner_r	- The radius of the inner circle.
    Output:
        outer_y	- y-coordinate of the outer circle centre.
        outer_x	- x-coordinate of the outer circle centre.
        outer_r	- Radius of the outer circle.
    """
    # Maximum displacement 15# (Daugman 2004)
    maxdispl = np.round(inner_r*0.15).astype(int)

    # 0.1 - 0.8 (Daugman 2004)
    minrad = np.round(inner_r/0.8).astype(int)
    maxrad = np.round(inner_r/0.3).astype(int)

    # # Hough Space (y,x,r)
    # hs = np.zeros([2*maxdispl, 2*maxdispl, maxrad-minrad])

    # Integration region, avoiding eyelids
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    # Resolution of the circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur
    sm = 7 	# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r


#------------------------------------------------------------------------------
def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    """
    Description:
        Performs contour (circular) integral.
    Input:
        imagen  - The input iris image.
        y_0     - The y-coordinate of the circle centre.
        x_0     - The x-coordinate of the circle centre.
        r       - The radius of the circle.
        angs    - The region of the circle considering clockwise 0-2pi.
    Output:
        hs      - Integral result.
    """
    # Get y, x
    # print(len(angs), r.shape[0], r.shape[1], r.shape[2])
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int) 
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 + np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)
    
    # Adapt y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1

    # Adapt x
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1


    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)

def process_for_daugman(self,IMG_PATH):
    # img = io.imread(IMG_PATH)
    # # lower,upper = calculate_thresholds(img)
    # img = rgb2gray(img)
    # lower = .04
    # upper = .025

    # # print("Found automatic threshold t = {}.".format(t))
    # # show_images([img], ['image in gray scale'])
    #     # apply automatic Canny edge detection using the computed median
    # detector = self.cannyEdgeDetector([img], sigma=4, kernel_size=5, lowthreshold=lower, highthreshold=upper, weak_pixel=255, strong_pixel=40)
    # imgs_final,keep = detector.detect()
    # # visualize(imgs_final, 'gray')
    # hough,outer_circles = self.hought_transform(keep,IMG_PATH,False,rmin=57,rmax=58,steps=800,threshold=.002)

    # lower = .054
    # upper  = .044
    # detector = self.cannyEdgeDetector([img], sigma=10, kernel_size=8, lowthreshold=lower, highthreshold=upper, weak_pixel=200, strong_pixel=10)
    # imgs_final,keep = detector.detect()
    # hough,inner_circles=self.hought_transform(keep,hough,True,rmin=30,rmax=40,steps=8000,threshold=.0005)
    # return (outer_circles,inner_circles)
    # IMG_PATH = 'images/tanwnl3.bmp'   
    img = io.imread(IMG_PATH,0)
    img = rgb2gray(img)
    output_image = Image.new("RGB", Image.open(IMG_PATH).size)
    output_image.paste(Image.open(IMG_PATH))
    i_y,i_x,i_r = self.searchInnerBound(img)
    draw_result = ImageDraw.Draw(output_image)
    draw_result.ellipse((i_x-i_r, i_y-i_r, i_x+i_r, i_y+i_r), outline=(255,0,0))
    o_y,o_x,o_r = self.searchOuterBound(img, i_y, i_x, i_r)
    draw_result.ellipse((o_x-o_r, o_y-o_r, o_x+o_r, o_y+o_r), outline=(0,255,0))
    output_image.save('images/ss.bmp')
    plt.imshow(output_image)
    plt.show()
    outer_circle=[]
    inner_circle=[]
    outer_circle.append((o_x,o_y,o_r))
    inner_circle.append((i_x,i_y,i_r))
    rowp = np.round(i_y).astype(int)
    colp = np.round(i_x).astype(int)
    rp = np.round(i_r).astype(int)
    row = np.round(o_y).astype(int)
    col = np.round(o_x).astype(int)
    r = np.round(o_r).astype(int)

    # Find top and bottom eyelid
    imsz = img.shape
    irl = np.round(row - r).astype(int)
    iru = np.round(row + r).astype(int)
    icl = np.round(col - r).astype(int)
    icu = np.round(col + r).astype(int)
    if irl < 0:
        irl = 0
    if icl < 0:
        icl = 0
    if iru >= imsz[0]:
        iru = imsz[0] - 1
    if icu >= imsz[1]:
        icu = imsz[1] - 1
    imageiris = img[irl: iru + 1, icl: icu + 1]
    return (outer_circle,inner_circle,imageiris)

def hought_transform(image,img,inner,rmin,rmax,steps,threshold):
    """
    This function applies the hough transform to an image.
    The function takes as input:
    - image: the image to be transformed
    - rmin: the minimum radius to be considered
    - rmax: the maximum radius to be considered
    - steps: the number of steps between rmin and rmax
    - threshold: the minimum number of votes that a line has to get in order to be considered
    """
    if not(inner) :
        output_image = Image.new("RGB", Image.open(IMG_PATH).size)
        output_image.paste(Image.open(IMG_PATH))
    else:
        output_image = Image.new("RGB", Image.open('images/hough_transform.bmp').size)
        output_image.paste(Image.open('images/hough_transform.bmp'))
    draw_result = ImageDraw.Draw(output_image)
    points=[]
    for r in (rmin,rmax+1):
        for t in range(steps):
            points.append((r,int(r*cos(2*pi*t/steps)),int(r*sin(2*pi*t/steps))))
    
    hough_space = defaultdict(int)
    for x, y in image:
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            hough_space[(a, b, r)] += 1
    circles=[]
    if not(inner):
        for k, v in sorted(hough_space.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / steps >= threshold and all((x - xc) * 2 + (y - yc) * 2 > rc ** 2 for xc, yc, rc in circles):
                # print(v / steps, x, y, r)
                circles.append((x, y, r))
    else:
        circles.append((132,131, 26))

    for x, y, r in circles:
        if not(inner):
            draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))
        else:
            draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(0,255,0,0))
    output_image.save('images/hough_transform.bmp')
    return (output_image,circles)