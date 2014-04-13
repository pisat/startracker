import csv
from math import *

class Point:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

class Polar:
	def __init__(self, r, theta, vphi):
		self.r = r
		self.theta = theta  # 0 < theta < pi
		self.vphi = vphi    # -pi < vphi < pi


def get_polar_from(point):
	r = sqrt(point.x**2 + point.y**2 + point.z**2)
	theta = acos(point.z/r)
	varphi = atan2(point.y,point.x)
	return Polar(r, theta, varphi)

def get_data_from_csv(filename):
	points = []
	with open(filename, 'rb') as f:
		f.readline()
		reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
		for row in reader:
			try:
				data = row[17:20]
				data = map(float,data)
			except:
				pass

			points.append(Point(*data))

	return points

cartesian_stars = get_data_from_csv('hygxyz_bigger9.csv')
polar_stars = map(get_polar_from, cartesian_stars)
stereographic_stars = map(lambda x: [1/(tan(x.theta)*2), x.vphi], polar_stars) # R, phi
normalized_cartesian = map(lambda x: Point(x.x/10000000,x.y/10000000,x.z/10000000), cartesian_stars)

import matplotlib.pyplot as plt
import pylab

a1 = map(lambda x: x.theta, polar_stars)
a2 = map(lambda x: x.vphi, polar_stars)
plt.clf()
plt.scatter(a1, a2, s=0.01)
plt.show()
plt.clf()

plt.scatter(a1, a2, s=0.01) 
F = pylab.gcf() 
F.patch.set_facecolor('black')
DPI = F.get_dpi()
DefaultSize = F.get_size_inches()
F.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
F.savefig("s1.eps")

plt.clf()
fig = plt.figure(figsize=(10,5),dpi=300,facecolor='black')
fig.subplots_adjust(wspace=.001,hspace=.001,left=.001,bottom=.001)
ax = fig.add_subplot(1,1,1,axisbg='black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.scatter(a1, a2, s=0.01, color='white', linewidth=0)

fig.patch.set_visible(False)
ax.axis('off')

with open('starfield.eps', 'w') as outfile:
    fig.canvas.print_eps(outfile, dpi=300)

plt.savefig("scatter.eps", facecolor=fig.get_facecolor(), transparent=True)
