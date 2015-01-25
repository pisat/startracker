# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Fast Linear Approximate Nearest Neighbour feature detection
# ===========================================================
# 
# Two main types of star tracking:
# 
# 1. Relative rotation and rate measurement
# 
# 2. The lost-in-space problem
# 
# We've chosen the lost-in-space problem as it's more interesting and more difficult.
# 
# 
# Okay, so to find stars in an image is an easy process, we can take a picture and produce some threshold values, such that anything darker is space and anything lighter is probably a star.
# 
# 
# 
# Using OpenCV and Fast Library Approximate Nearest Neighbour (FLANN) to search for matches.
# 
# This code uses Python **2.7.3** and OpenCV **3.0.0**

# <codecell>

import numpy as np
import cv2
from matplotlib import pyplot as plt

# <markdowncell>

# We'll Import our image to use as an example.

# <codecell>

sample_image = cv2.imread('star.jpg')
plt.figure(figsize(12,12))
plt.imshow(sample_image)

# <markdowncell>

# We can now take a threshold value for this image, in this case, the average luminosity of the image.

# <codecell>

gray = cv2.cvtColor(sample_image,cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
plt.imshow(th1,'gray')

# <markdowncell>

# So without much effort, we've made it extremely easy to detect stars in the night sky from the position of white cells in the image.
# This only solves the easy part of the problem, to find the position of a star field is without the celestial mackdro f.

# <codecell>


img1 = cv2.imread('star_crop.jpg',0)           # queryImage
#img1 = cv2.imread('star_crop_distort.jpg',0) #The Distorted image
img2 = cv2.imread('star.jpg',0) # trainImage
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.figure(figsize(10,10))
plt.imshow(img3,),plt.show()

# <markdowncell>

# However, sift contains position information and is comparatively slow, we'll use SURF instead. With invariant direction, only size as stars should look the same from any angle (obviously).

# <codecell>

surf = cv2.SURF(10)
surf.upright = True #Direction invariant
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
#for kps in kp1:
#    print "x: " + str(kps.pt[0]) + " y: " + str(kps.pt[1]) + " Size: " + str(kps.size) + " Octave: " \
#    + str(kps.octave) + " Response: " + str(kps.response)
#for desc in des1:
#    print desc
#    break
print len(kp1)

# <codecell>

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.figure(figsize(10,10))
plt.imshow(img3,),plt.show()

# <markdowncell>

# Let's use this to give us a relative position, if we assume the middle of the picture is 0az (lat), 0zen(lon).
# We'll assume this picture gives us a full view of the sky.

# <codecell>

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        selected_match = m
        print "Destination x:" + str(kp1[m.queryIdx].pt[0] ) + "  y:" + str(kp1[m.queryIdx].pt[1] )
        print "Sky x:" + str(kp2[m.trainIdx].pt[0] ) + "  y:" + str(kp2[m.trainIdx].pt[1] )
        break #Just so we pick one point and 

# <markdowncell>

# Get the centroid of the image and relate it to a keypoint.

# <codecell>

print "Destination x:" + str(kp1[m.queryIdx].pt[0] ) + "  y:" + str(kp1[m.queryIdx].pt[1] )
print "Sky x:" + str(kp2[m.trainIdx].pt[0] ) + "  y:" + str(kp2[m.trainIdx].pt[1] )
height, width = img1.shape
height2, width2 = img2.shape
print height2
print width2
cent_x = kp1[selected_match.queryIdx].pt[0] - width/2
cent_y = kp1[selected_match.queryIdx].pt[1] - height/2
print "Offset x: " + str(cent_x ) + " y:" + str(cent_y)

# <markdowncell>

# To the center of the main image

# <codecell>

#print kp2[selected_match.trainIdx].pt[0]
#print kp2[selected_match.trainIdx].pt[1]
azimuth = 180*((kp2[selected_match.trainIdx].pt[0] - cent_x)-(width2/2))/(width2/2)
zenith = 180*((height2/2)-(kp2[selected_match.trainIdx].pt[1] - cent_y))/(height2/2)
print "Azimuth angle in the sky: " + str(azimuth)
print "Zenith angle in the sky: " + str(zenith)

# <markdowncell>

# Now we want to do this for the whole starfield

# <codecell>

import csv
from math import *

class Point:
	def __init__(self, x, y, z, mag=0):
		self.x = x
		self.y = y
		self.z = z
                self.mag = mag

class Polar:
	def __init__(self, r, theta, vphi, mag=0):
		self.r = r
		self.theta = theta  # 0 < theta < pi
		self.vphi = vphi    # -pi < vphi < pi
                self.mag = mag


def get_polar_from(point):
	r = sqrt(point.x**2 + point.y**2 + point.z**2)
	theta = acos(point.z/r)
	varphi = atan2(point.y,point.x)
	return Polar(r, theta, varphi, point.mag)

def get_data_from_csv(filename):
	points = []
	with open(filename, 'rb') as f:
		f.readline()
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
		for row in reader:
			try:
				data = row[17:20]
                                data.append(row[13])
				data = map(float,data)
			except:
				pass

			points.append(Point(*data))

	return points

cartesian_stars = get_data_from_csv('hygxyz_bigger9.csv')
polar_stars = map(get_polar_from, cartesian_stars)

a1 = map(lambda x: x.theta, polar_stars)
a2 = map(lambda x: x.vphi, polar_stars)
mag = map(lambda x: x.mag, polar_stars)

keyPointsFromMap = map(lambda x: cv2.KeyPoint(x.theta,x.vphi,x.mag),polar_stars)

# <codecell>

print polar_stars[0].mag

# <codecell>

plt.scatter(a1, a2, s=0.01)


# <markdowncell>

# This looks very like other images designed to map the same space, e.g. [This code](http://chrusion.com/public_files/tycho/Stars_Tycho2_3000.png) and [this site](http://paulbourke.net/miscellaneous/starfield/)
# 
# Let's reproduce the stars more realistically. We've got useful code for this from [Chris Frohmaier](https://github.com/chrisfrohmaier/Ben_Space_App) and [Jon Sowman](http://nbviewer.ipython.org/url/u.jonsowman.com/Ben_Space_App.ipynb)

# <codecell>


# <codecell>

import random
ysize=3000
xsize=1500
scale=10
x,y = np.indices((xsize,ysize))
img_noise = np.random.normal(scale=0.000001, size=(xsize,ysize)) #+ 4.321 #G
img_clear = np.zeros((xsize,ysize))
render_img = img_noise.copy()
mean=2.5
sigma=1 #suggested to be 0.4
flux_value = lambda magn: 2.512**(-magn)
# Just a Gaussian function. 
# This is the PSF of a star, all stars have a gaussian function which varies with the quality 
# of your observing instrument/conditions
make_star = lambda x0, y0, amp: flux_value(amp) * np.exp(-0.5/sigma**2*((x[x0-scale:x0+scale,y0-scale:y0+scale]-x0)**2 + (y[x0-scale:x0+scale,y0-scale:y0+scale]-y0)**2))

#img = img_noise.copy()
i = 1;
xfac = (xsize/np.pi)
yfac = ysize/(2*np.pi)
for s in polar_stars:
    #Just do the first 10000 for now as this takes ages!
    if (i % 10000==0):
        print("Making star {} at ({:.2f},{:.2f}) with peak magnitude {:.2f}".format(i, s.theta, s.vphi, s.mag))
        #break
    i = i+1
    xn = s.theta*xfac
    yn = (s.vphi+np.pi)*yfac
    render_img[xn-scale:xn+scale,yn-scale:yn+scale] += make_star(xn, yn, s.mag-25)
fig, ax = pylab.subplots(figsize=(16,16))
# Change interpolation to 'nearest' for a pure pixel map. 'Bicubic' smooths it out (blurs). 
# Read numpy manual for interpolation meanings
final_image=plt.imshow(render_img, interpolation='nearest') 
final_image.set_cmap('gray')
plt.show()


# <markdowncell>

# Now that we can print the start grid, we can use sift on this!

# <codecell>

# Initiate SIFT detector
sift = cv2.SURF(2)

overlay_img = render_img.copy()
#surf2 = cv2.SURF(10)
#surf2.upright = True #Direction invariant
plt.imsave('temp.jpg',render_img)
temp_img = cv2.imread('temp.jpg',0)
render_kps, render_descriptors = sift.detectAndCompute(temp_img,None)

cv2.drawKeypoints(temp_img,render_kps, overlay_img)
print len(render_kps)
plt.imshow(overlay_img)

# <codecell>

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann2 = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann2.knnMatch(render_descriptors,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img4 = cv2.drawMatchesKnn(overlay_img,render_kps,img2,kp2,matches,None,**draw_params)
plt.figure(figsize(10,10))
plt.imshow(img4,),plt.show()

# <markdowncell>

# And... the result is that nothing has come out, no matches. However at the moment the two images are nothing alike, they could do with some editing to make them more similar. On top of which, the kth nearest searching algorithm looks at the descriptors of the keypoints which are not very useful here but not the relative position of nodes to eachother.

# <codecell>

def getDescriptor(image, keyPoint)
    return img[keyPoint
    

# <codecell>


# <codecell>

size = 2000
pi = 3.14159265
starfield = np.zeros([size,size],dtype=numpy.float)
starfield_plain = np.zeros([size,size],dtype=numpy.float)
for i in xrange(1,len(a1)):
    xtar = int(a1[i]*size/pi)
    ytar = int((pi+a2[i])*0.5*size/pi)
    #for j in range(xtar-1,xtar+1):
    #    for k in range(ytar-1,ytar+1):
    #        starfield[j,k] += 2.5*mag[i]
    starfield_plain[xtar,ytar]+=mag[i]*numpy.power(10,15)*numpy.power(10,15)#numpy.power(10,(-2.5*mag[i]))
    starfield[xtar,ytar]+=numpy.power(10,(-2.5*mag[i]))
    #print starfield[int(a1[i]*size/pi),int((pi+a2[i])*0.5*size/pi)]
    #print [int(a1[i]*size/pi),int((pi+a2[i])*0.5*size/pi)],starfield[int(a1[i]*size/pi),int((pi+a2[i])*0.5*size/pi)]

# <codecell>

starfield_plain2 = Image.fromarray(starfield_plain)#Works well with Image.fromarray(starfield,'I') or 'F'
plt.imshow(starfield_plain2)
starfield_plainrgb = starfield_plain2.convert('RGB')
starfield_plainrgb.save("starfield_plain.jpg")
FileLink('starfield_plain.jpg')

# <codecell>

print np.amax(starfield), np.amin(starfield)
#Normalise array
starfield = np.log(starfield*(1/np.amax(starfield))+1)
print np.amax(starfield), np.amin(starfield)

# <codecell>

from PIL import Image
import PIL
from IPython.display import FileLink, FileLinks
starfield2 = Image.fromarray(numpy.power(10,15)*numpy.power(10,10)*starfield)#Works well with Image.fromarray(starfield,'I') or 'F'
plt.imshow(starfield2)
starfieldrgb = starfield2.convert('RGB')
starfieldrgb.save("starfield2.jpg")
FileLink('starfield2.jpg')

# <codecell>

starfield3 = starfield2.resize((size/10,size/10), PIL.Image.ANTIALIAS)
plt.imshow(starfield3)

# <codecell>




import pylab
fig = pylab.gcf()
starfield = fig.canvas.print_raw

# <codecell>

from cStringIO import StringIO
sio=StringIO()
fig.canvas.print_png(sio)

img2 = cv2.imdecode(np.asarray(bytearray(sio),dtype=np.uint8),-1)
plt.imshow(img2)

# <codecell>

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.figure(figsize(10,10))
plt.imshow(img3,),plt.show()

