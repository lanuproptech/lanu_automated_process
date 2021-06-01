import json
import pickle

for t in range(100):
    print(t)

    with open('../tileSections/Tile_-20_7_section_' + str(t) + '.json') as json_file:
        dictData = json.load(json_file)
    keys = list(dictData.keys())
    points = dictData['features']

    Pts = []
    Normals = []
    Ele = []
    for p in points:
        xy_point = p['geometry']['coordinates']
        h = p['properties']['RoofData']['Height']
        Pts.append([xy_point[0], xy_point[1], h])
        n_dict = p['properties']['RoofData']['Normals']
        Normals.append([n_dict['x'], n_dict['y'], n_dict['z']])
        Ele.append(p['properties']['TerrainData']['Elevation'])

    save_pts = '../tileSections/Tile_-20_7_section_pts_' + str(t) +'.pickle'
    save_normals = '../tileSections/Tile_-20_7_section_normals_' + str(t) +'.pickle'
    save_ele = '../tileSections/Tile_-20_7_section_ele_' + str(t) +'.pickle'

    with open(save_pts, 'wb') as handle:
        pickle.dump(Pts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_normals, 'wb') as handle:
        pickle.dump(Normals, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_ele, 'wb') as handle:
        pickle.dump(Ele, handle, protocol=pickle.HIGHEST_PROTOCOL)