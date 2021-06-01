import os
import pickle
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageChops
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
import colorsys
import scipy.linalg as la

def get_aspect_ratio_area(x, y):
    M = np.zeros((2, 2))
    cx = sum(x) / len(x)
    cy = sum(y) / len(y)
    area = 0
    for i in range(len(x)):
        i1 = (i + 1) % len(x)

        area += 0.5 * abs((x[i1] - cx) * (y[i] - cy) - (x[i] - cx) * (y[i1] - cy))
        ix = x[i] - cx
        iy = y[i] - cy
        M[0][0] += ix ** 2
        M[1][1] += iy ** 2
        M[0][1] -= ix * iy
        M[1][0] -= ix * iy

    eig = la.eig(M)[0]
    evalues = [eig[0].real, eig[1].real]
    evalues = [abs(i) for i in evalues]
    if min(evalues) > 0:
        aspect_ratio = np.sqrt(max(evalues) / min(evalues))
    else:
        aspect_ratio = -1
    area = round(100 * area) / 100

    return aspect_ratio, area

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def area_polygon(x, y):
    cx = sum(x) / max(1, len(x))
    cy = sum(y) / max(1, len(y))
    area = 0
    for i in range(len(x)):
        i1 = (i - 1) % len(x)
        area += 0.5 * ((x[i1] - cx) * (y[i] - cy) - (x[i] - cx) * (y[i1] - cy))

    area = round(100 * area) / 100
    return area

def point_in_polygon(px, py, x, y):
    # returns True if point (px, py) is in polygon (x, y) and False otherwise
    x0, y0 = x[:], y[:]
    c = False
    n = len(x0)
    for i in range(n):
        j = (i - 1) % len(x0)
        if (((y0[i] > py) != (y0[j] > py)) and (
                px >= ((x0[j] - x0[i]) * (py - y0[i]) / (y0[j] - y0[i])) + x0[i])):
            c = not c
    return c

def trim_alt(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    # Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    # If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def crop_image(im_str):
    im = Image.open(im_str)
    im = trim_alt(im)
    im.show()
    # im.save(im_str)
    return

def crop_and_resize_image(self, im, h):
    height = h
    image = Image.open(im)
    image = self.trim(image)

    hpercent = (height / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    image = image.resize((wsize, height), PIL.Image.ANTIALIAS)
    old_size = image.size
    new_size = (old_size[0] + 400, old_size[1] + 400)
    new_image = Image.new("RGB", new_size, color=(255, 255, 255))  ## luckily, this is already black!
    new_image.paste(image, (int((new_size[0] - old_size[0]) / 2), int((new_size[1] - old_size[1]) / 2)))
    new_image.save(im)

    return

top_dir = '/Users/lukecoburn/Dropbox/Lanu/houses/test_height_data'
houses = [x for x in os.walk(top_dir) ] #if 'ynmouth' in x]
houses = houses[0][2]
houses = [x for x in houses if '.png' in x]

bad_files = ['1_Lynmouth_Dr_Ruislip_HA4_9BZ_UK.png','2_Lynmouth_Dr_Ruislip_HA4_9BZ_UK.png','8_Lynmouth_Dr_Ruislip_HA4_9BZ_UK.png','80_Lynmouth_Dr_Ruislip_HA4_9BZ_UK.png']

houses = houses
count = 0
for file in [houses[0]]:
    count += 1
    print(count, len(houses))
    if file not in bad_files:
        plt.close('all')
        print('')
        name = os.path.splitext(file)[0]
        png_file = top_dir + '/' + file
        position_file = top_dir + '/' + name+'_position.pickle'
        site_position_file = top_dir + '/' + name+'_site_position.pickle'
        data_file = top_dir + '/' + name+'_data.pickle'

        # Load position data and height data
        with open(position_file, 'rb') as f:
            position = pickle.load(f)
        with open(site_position_file, 'rb') as f:
            site_position = pickle.load(f)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        x = position['x']
        y = position['y']
        house_vector = [x[1]-x[0], y[1]-y[0]]
        len_hv = np.sqrt(house_vector[0]**2 + house_vector[1]**2)
        house_vector = [house_vector[0]/len_hv, house_vector[1]/len_hv]

        xs = site_position['xs']
        ys = site_position['ys']

        x_ = data['x_']
        y_ = data['y_']
        zl_ = data['zl_']
        zu_ = data['zu_']
        u_ = data['u_']
        v_ = data['v_']
        w_ = data['w_']
        xf_ = data['xf_']
        yf_ = data['yf_']
        zf_ = data['zf_']
        uf_ = data['uf_']
        vf_ = data['vf_']
        wf_ = data['wf_']
        x_ += xf_
        y_ += yf_
        zu_ += zf_
        zl_ += [sum(zl_)/len(zl_)]*len(zf_)
        u_ += uf_
        v_ += vf_
        w_ += wf_
        hw, hd = max(x_) - min(x_), max(y_) - min(y_)

        # open and crop image
        num_cluster = 10
        image = cv2.imread(png_file)

        iw = image.shape[1]
        ih = image.shape[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to be a list of pixels
        image_reshape = image.reshape((image.shape[0] * image.shape[1], 3))

        # cluster the pixel intensities
        clt = KMeans(n_clusters=num_cluster)
        clt.fit(image_reshape)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = centroid_histogram(clt)
        bar = plot_colors(hist, clt.cluster_centers_)

        hsv_colors = []
        rgb_colors = []
        for c in clt.cluster_centers_:
            h, s, v = colorsys.rgb_to_hsv(c[0]/ 255, c[1]/ 255, c[2]/ 255)
            h = int(h * 360)
            s = int(s * 100)
            v = int(v * 100)
            hsv_colors.append([h,s,v])
            rgb_colors.append([int(c[0]), int(c[1]), int(c[2])])

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        delta = 20
        X, Y = [], []
        for c in rgb_colors:
            c_lower = (max(c[0]-delta,0), max(c[1]-delta,0), max(c[2]-delta,0))
            c_upper = (min(c[0]+delta,255), min(c[1]+delta,255), min(c[2]+delta,255))
            mask = cv2.inRange(image, c_lower, c_upper)
            ## final mask and masked
            target = cv2.bitwise_and(image, image, mask=mask)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            len(contours)
            base = target.copy()
            contour_img = cv2.drawContours(base, contours, -1, (255,0,0), 2)

            borderx, bordery = 0, 0
            for c in contours:
                x_vec, y_vec = [], []
                for j in range(len(c)):
                    x_temp = c[j][0][0]
                    y_temp = c[j][0][1]
                    x_temp = hw*(c[j][0][0] + borderx)/iw
                    y_temp = hd*(ih - (c[j][0][1] - bordery))/ih
                    x_vec.append(x_temp)
                    y_vec.append(y_temp)

            X.append(x_vec)
            Y.append(y_vec)

        A, FA, AR = [],[],[]
        Total_Area = hw*hd
        for i in range(len(X)):
            area = area_polygon(X[i], Y[i])
            aspect_ratio, area = get_aspect_ratio_area(X[i], Y[i])
            A.append(round(area,3))
            FA.append(round(area/Total_Area,3))
            AR.append(round(aspect_ratio,3))

        U, V, W, LV, PU, PV = [-10]*len(X), [-10]*len(X), [-10]*len(X), [-10]*len(X), [-10]*len(X), [-10]*len(X)
        HL, HU_min, HU_max = [0]*len(X), [0]*len(X), [0]*len(X)
        Vertical = [False]*len(X)

        for i in range(len(X)):
            if A[i] > 0.5:
                u_temp = 0
                v_temp = 0
                w_temp = 0
                hu_temp = []
                hl_temp = []
                n = 0
                for j in range(len(x_)):
                    if point_in_polygon(x_[j], y_[j], X[i], Y[i]):
                        u_temp += u_[j]
                        v_temp += v_[j]
                        w_temp += w_[j]
                        hu_temp.append(zu_[j])
                        hl_temp.append(zl_[j])
                        n += 1
                U[i] = round(u_temp/max(1,n),3)
                V[i] = round(v_temp/max(1,n),3)
                W[i] = round(w_temp/max(1,n),3)
                PU[i] = U[i]
                PV[i] = V[i]
                len_p = np.sqrt(U[i]**2 + V[i]**2)
                if len_p > 0:
                    PU[i] = round(PU[i]/len_p,3)
                    PV[i] = round(PV[i]/len_p,3)
                LV[i] = round(np.sqrt(U[i]**2 + V[i]**2 + W[i]**2),3)
                HL[i] = round(sum(hl_temp)/max(1,len(hl_temp)),3)
                if len(hu_temp) > 0:
                    HU_min[i] = min(hu_temp)
                    HU_max[i] = max(hu_temp)
                else:
                    HU_min[i] = -10
                    HU_max[i] = -10

                if abs(V[i]) > 0.9:
                    Vertical[i] = True

        for i in range(len(A)-1):
            for j in range(i+1, len(A)):
                if A[j] == A[i]:
                    A[j] = 0
                    AR[j] = 0
                    FA[j] = 0
                    U[j] = -10
                    V[j] = -10
                    W[j] = -10
                    PU[j] = -10
                    PV[j] = -10
                    HL[j] = -10
                    HU_max[j] = -10
                    HU_min[j] = -10
                    LV[j] = -10

        dict = {}
        dict['x'] = x
        dict['y'] = y
        dict['house_vector'] = house_vector
        dict['xs'] = xs
        dict['ys'] = ys
        dict['X'] = X
        dict['Y'] = Y
        dict['U'] = U
        dict['V'] = V
        dict['W'] = W
        dict['PU'] = PU
        dict['PV'] = PV
        dict['HL'] = HL
        dict['HU_max'] = HU_max
        dict['HU_min'] = HU_min
        dict['LV'] = LV
        dict['Vertical'] = Vertical
        dict['A'] = A
        dict['FA'] = FA
        dict['AR'] = AR

        save_data = top_dir + '/' + name+'_summary.pickle'
        with open(save_data, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)