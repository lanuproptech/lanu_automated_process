from pathlib import Path
import sys
import os
from os import path
home = str(Path.home())
sys.path.append(path.abspath(home))
import json
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometryTools.geometry import Geometry
from lanu_reader.reader import Reader, Box, Block
import numpy as np
import math

def plot_face(x, y, z, color, edge_color, alpha, lw, ax):

    z = [max(i, 0) for i in z]
    verts = [list(zip(x, y, z))]

    collection = Poly3DCollection(verts, linewidths=lw, edgecolors=edge_color, alpha=alpha)
    collection.set_facecolor(color)
    ax.add_collection3d(collection)

    return ax

def rotate_by_angle(x, y, alpha):
    cx_ = sum(x)/len(x)
    cy_ = sum(y)/len(y)
    if type(x) == list:
        if len(x) > 0:
            x_ = x[:]
            y_ = y[:]
            for i in range(len(x)):
                x[i] = round(((x_[i] - cx_) * np.cos(alpha) - (y_[i] - cy_) * np.sin(alpha)), 4)
                y[i] = round(((x_[i] - cx_) * np.sin(alpha) + (y_[i] - cy_) * np.cos(alpha)), 4)
    else:
        x_ = x
        y_ = y
        x = round(x_ * np.cos(alpha) - y_ * np.sin(alpha), 4)
        y = round(x_ * np.sin(alpha) + y_ * np.cos(alpha), 4)

    return x, y

def load_tile(main):

    with open(home + '/tileSections/Tile_bounds.pickle', 'rb') as f:
        Tile = pickle.load(f)

    tile_ind = -1
    centre = main.house.gf.centre
    for count, tile in enumerate(Tile):
        px, py = centre[0], centre[1]
        x, y = tile[0], tile[1]
        if gt.point_in_polygon(px, py, x, y):
            tile_ind = count
    print(tile_ind, centre, Tile[tile_ind])

    with open(home + '/tileSections/Tile_-20_7_section_' + str(tile_ind) + '.json') as json_file:
        tile = json.load(json_file)

    return tile

home = str(Path.home())


houses = [x[0] for x in os.walk(home + '/Dropbox/Lanu/houses/') if '67_Lynmouth' in x[0]]

count = 0
for house_name in [houses[0]]:

    plt.close('all')
    count += 1
    print('*', count, house_name)
    gt = Geometry()

    # Import data from databases
    home = str(Path.home())
    r = Reader(house_name, gt)
    r.house_name = house_name
    gt.house_name = house_name
    r.get_data_and_points_from_db()
    plot_bool =  True
    plot_color_bool = True
    move_bool = True
    points_bool = True
    plot_PD = True
    main_pd, main_pp, left, right = r.run_case(r, plot_bool, plot_color_bool, move_bool, points_bool, plot_PD)

"""
    main = main_pd
    house_model = {}
    x, y = main.house.gf.x, main.house.gf.y
    # X_,Y_,Z_, faces, pts, normals= gt.basic_model_from_height_data(x, y, plot_bool)
    x_, y_, zl_, zu_ = main.house.pts[0], main.house.pts[1], main.house.pts[2], main.house.pts[3]
    u_, v_, w_ = main.house.normals[0], main.house.normals[1], main.house.normals[2]
    house_model['main'] = [main.house.X_shape[0], main.house.Y_shape[0], main.house.Z_shape[0]]
    house_model['roof'] = [main.house.X_shape[1], main.house.Y_shape[1], main.house.Z_shape[1]]
    house_model['faces'] = main.house.faces

    b = [main.house.X_shape, main.house.Y_shape, main.house.Z_shape, main.house.faces, main.house.pts, main.house.normals]

    pickle_out = open("vector_field.pickle", "wb")
    pickle.dump(b, pickle_out)
    pickle_out.close()

    filename = house_name + '/' + os.path.basename(house_name) + '_simple_model.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(house_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    okeys = list(main.house.original.keys())
    for k in okeys:
        if len(k) == 1:
            x, y = main.house.original[k].x, main.house.original[k].y
            X_,Y_,Z_, faces = gt.basic_model_from_height_data(x, y)
            str1 = 'O' + k
            house_model[str1] = [X_[0],Y_[0],Z_[0]]
            if len(X_) == 2:
                house_model[str1+'_roof'] = [X_[1],Y_[1],Z_[1]]

    ekeys = list(main.house.existing.keys())
    for k in ekeys:
        if len(k) == 1:
            x, y = main.house.existing[k].x, main.house.existing[k].y
            str1 = 'O' + k
            X_,Y_,Z_, faces = gt.basic_model_from_height_data(x, y)
            house_model['str1'] = [X_,Y_,Z_]

    pkeys = list(house_model)
    for k in pkeys:
        if k != 'faces':
            X = house_model[k][0]
            Y = house_model[k][1]
            Z = house_model[k][2]
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
            ax = gt.plot_XYZ_block(faces, X, Y, Z,  'r', 'k', 0.25, 2, ax)

ax.set_xlim3d(min_x-2, max_x+2)
ax.set_ylim3d(min_y-2, max_y+2)
ax.set_zlim3d(min_z-2, max_z+2)
"""