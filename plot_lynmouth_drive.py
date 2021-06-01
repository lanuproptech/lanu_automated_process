import os
import pickle
from pathlib import Path
import sys
from os import path
home = str(Path.home())
sys.path.append(path.abspath(home))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import numpy as np

from geometryTools.geometry import Geometry

def max_min_x_y(X, Y, Z, min_x, min_y, min_z, max_x, max_y, max_z):
    if min(X) < min_x:
        min_x = min(X)
    if min(Y) < min_y:
        min_y = min(Y)
    if min(Z) < min_z:
        min_z = min(Z)
    if max(X) > max_x:
        max_x = max(X)
    if max(Y) > max_y:
        max_y = max(Y)
    if max(Z) > max_z:
        max_z = max(Z)
    return min_x, min_y, min_z, max_x, max_y, max_z

gt = Geometry()

top_dir = '/Users/lukecoburn/Dropbox/Lanu/houses/test_height_data'
houses = [x for x in os.walk(top_dir) ]#if 'ynmouth' in x]
houses = houses[0][2]
houses = [x for x in houses if '_summary.pickle' in x]

min_x, max_x, min_y, max_y, min_z, max_z = 100000000, -100000000, 100000000, -100000000, 100000000, -100000000
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(elev=30., azim=-174.)
ax.axis("off")

houses = houses
count = 0
count_dormer = 0
count_g = 0
count_hl = 0
count_hr = 0
for file in houses:
    # try:
        print(file)
        count += 1
        name = os.path.splitext(file)[0]
        data_file = top_dir + '/' + name + '.pickle'
        png_file = top_dir + '/' + name[:-8] + '.png'

        # Load position data and height data
        with open(data_file, 'rb') as f:
            dict = pickle.load(f)
        x = dict['x']
        y = dict['y']
        house_vector = dict['house_vector']
        xs = dict['xs']
        ys = dict['ys']
        XB = dict['X']
        YB = dict['Y']
        U = dict['U']
        V = dict['V']
        W = dict['W']
        PU = dict['PU']
        PV = dict['PV']
        HL = dict['HL']
        HU_max = dict['HU_max']
        HU_min = dict['HU_min']
        LV = dict['LV']
        Vertical = dict['Vertical']
        A = dict['A']
        FA = dict['FA']

        # Roof
        dormer = [i for i in range(len(Vertical)) if Vertical[i] and FA[i] > 0.1]
        chimney = [i for i in range(len(Vertical)) if Vertical[i] and FA[i] < 0.1]
        pieces = [i for i in range(len(Vertical)) if not Vertical[i] and FA[i] > 0.2]

        if len(dormer) > 0:
            print('dormer', dormer)

        if len(pieces) > 0 or len(dormer) > 0:

            if len(dormer) > 0:
                col = 'r'
            else:
                col = 'b'

            # Plot house
            faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

            HL_temp = [x for x in HL if x > 0]
            HU_max = [x for x in HU_max if x > 0]
            HU_min = [x for x in HU_min if x > 0]
            zl_house = max(HL_temp)
            HU_min_temp = [x for x in HU_min if x > zl_house + 4]
            if len(pieces) > 0:
                zlr_house = HU_min[pieces[0]]
                zur_house = HU_max[pieces[0]]
            else:
                zlr_house = min(HU_min_temp)
                zur_house = max(HU_max)

            # Plot main block
            X = x + x
            Y = y + y
            Z = [zl_house]*4 + [zlr_house]*4
            ax = gt.plot_XYZ_block(faces, X, Y, Z, col, 'k', 0.25, 2, ax)
            min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(X, Y, Z, min_x, min_y, min_z, max_x, max_y, max_z)

            # Site
            XS = xs
            YS = ys
            ZS = [zl_house]*len(xs)
            ax = gt.plot_XYZ_block([list(range(0,len(xs)))], XS, YS, ZS, 'g', 'k', 0.125, 2, ax)
            min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(XS, YS, ZS, min_x, min_y, min_z, max_x, max_y, max_z)

            if len(pieces) == 2:
                roof_type = 'G'
            elif len(pieces) == 3:
                for i in pieces:
                    u = [1,0]
                    v = [PU[i], PV[i]]
                    if np.dot(u,v) > 0.9:
                        roof_type = 'HR'
                    elif np.dot(u,v) < -0.9:
                        roof_type = 'HL'
                    else:
                        roof_type = 'G'

            if roof_type == 'G':
                count_g += 1
                xr = [(x[0] + x[3])/2, (x[1] + x[2])/2, (x[1] + x[2])/2, (x[0] + x[3])/2]
                yr = [(y[0] + y[3])/2, (y[1] + y[2])/2, (y[1] + y[2])/2, (y[0] + y[3])/2]
                XR = x + xr
                YR = y + yr
                ZR = [zlr_house]*4 + [zur_house]*4
                ax = gt.plot_XYZ_block(faces, XR, YR, ZR,  col, 'k', 0.25, 2, ax)
                min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(XR, YR, ZR, min_x, min_y, min_z, max_x, max_y, max_z)

            elif roof_type == 'HR':
                count_hr += 1
                xr = [(x[0] + x[3])/2, 0.5*(x[1] + x[2])/2, 0.5*(x[1] + x[2])/2, (x[0] + x[3])/2]
                yr = [(y[0] + y[3])/2, 0.5*(y[1] + y[2])/2, 0.5*(y[1] + y[2])/2, (y[0] + y[3])/2]
                XR = x + xr
                YR = y + yr
                ZR = [zlr_house]*4 + [zur_house]*4
                ax = gt.plot_XYZ_block(faces, XR, YR, ZR,  col, 'k', 0.25, 2, ax)
                min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(XR, YR, ZR, min_x, min_y, min_z, max_x, max_y, max_z)

            elif roof_type == 'HL':
                count_hl += 1
                xr = [0.5*(x[0] + x[3])/2, (x[1] + x[2])/2, (x[1] + x[2])/2, 0.5*(x[0] + x[3])/2]
                yr = [0.5*(y[0] + y[3])/2, (y[1] + y[2])/2, (y[1] + y[2])/2, 0.5*(y[0] + y[3])/2]
                XR = x + xr
                YR = y + yr
                ZR = [zlr_house]*4 + [zur_house]*4
                ax = gt.plot_XYZ_block(faces, XR, YR, ZR,  col, 'k', 0.25, 2, ax)
                min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(XR, YR, ZR, min_x, min_y, min_z, max_x, max_y, max_z)

            if len(dormer) > 0:
                count_dormer += 1
                xd = [(x[0] + x[3])/2, (x[1] + x[2])/2, x[2], x[3]]
                yd = [(y[0] + y[3])/2, (y[1] + y[2])/2, y[2], y[3]]
                XD = xd + xd
                YD = yd + yd
                ZD = [zlr_house] * 4 + [zlr_house + 2.6] * 4
                ax = gt.plot_XYZ_block(faces, XD, YD, ZD,  col, 'k', 0.25, 2, ax)
                min_x, min_y, min_z, max_x, max_y, max_z = max_min_x_y(XD, YD, ZD, min_x, min_y, min_z, max_x, max_y, max_z)

            mx = 0.5*(max_x + min_x)
            my = 0.5*(max_y + min_y)
            mz = 0.5*(max_z + min_z)
            dx = max_x - min_x
            dy = max_y - min_y
            dz = max_z - min_z
            d = max(dx, dy, dz)

    # except:
    #     print('Missing house')

ax.set_xlim3d(mx - d / 2, mx + d / 2)
ax.set_ylim3d(my - d / 2, my + d / 2)
ax.set_zlim3d(mz - d / 2, mz + d / 2)