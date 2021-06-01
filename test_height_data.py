import sys

from pathlib import Path
from os import path
home = str(Path.home())
sys.path.append(path.abspath(home))

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle

from geometryTools.geometry import Geometry
from lanu_reader.reader import *

# define a global house geometry object
gt = Geometry()

# Import data from databases
home = str(Path.home())
r = Reader(home + '/Dropbox/Lanu/houses/67_Lynmouth_Dr_Ruislip_HA4_9BY_UK', gt)
plot_bool, plot_color_bool, move_bool, points_bool, plot_PD = True, False, True, False, True
main_pd, main_pp, left, right = r.run_case(r, plot_bool, plot_color_bool, move_bool, points_bool, plot_PD)

main = main_pd

with open(home + '/tileSections/Tile_bounds.pickle', 'rb') as f:
    Tile = pickle.load(f)

tile_ind = -1
centre = main.house.gf.centre
for count, tile in enumerate(Tile):
    px, py = centre[0], centre[1]
    x,y = tile[0], tile[1]
    if gt.point_in_polygon(px, py, x, y):
        tile_ind = count
print(tile_ind, centre, Tile[tile_ind])

with open(home + '/tileSections/Tile_-20_7_section_'+str(tile_ind)+'.json') as json_file:
    tile = json.load(json_file)

x, y = main.house.gf.x, main.house.gf.y
house_tile = []
pts = []
for i in range(len(tile['features'])):
    p = tile['features'][i]['geometry']['coordinates']
    px, py = p[0], p[1]
    if gt.point_in_polygon(px, py, x, y):
        house_tile.append(tile['features'][i])

Pts = []
Normals = []
Ele = []
for p in house_tile:
    xy_point = p['geometry']['coordinates']
    h = p['properties']['RoofData']['Height']
    Pts.append([xy_point[0], xy_point[1], h])
    n_dict = p['properties']['RoofData']['Normals']
    Normals.append([n_dict['x'], n_dict['y'], n_dict['z']])
    Ele.append(p['properties']['TerrainData']['Elevation'])

x_, y_, z_, u_, v_, w_ = [], [], [], [], [], []
xs_, ys_, zs_, us_, vs_, ws_ = [], [], [], [], [], []

for i in range(len(Pts)):

    px = Pts[i][0]
    py = Pts[i][1]
    if px > max_x:
        max_x = px
    if py > max_y:
        max_y = py
    if px < min_x:
        min_x = px
    if py < min_y:
        min_y = py
    if gt.point_in_polygon(px, py, x[:4], y[:4]):
        x_.append(Pts[i][0])
        y_.append(Pts[i][1])
        z_.append(Pts[i][2])
        zs_.append(Ele[i])
        if Normals[i][2] > 0:
            u_.append(Normals[i][0])
            v_.append(Normals[i][1])
            w_.append(Normals[i][2])
        else:
            u_.append(-Normals[i][0])
            v_.append(-Normals[i][1])
            w_.append(-Normals[i][2])

fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(elev=30., azim=-174.)
ax.axis("off")

ax.quiver(x_, y_, z_, u_, v_, w_, length=1, color ='b')
ax.plot(x_, y_, zs_, '.', color ='r')