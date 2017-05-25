import numpy as np
from PIL import Image
import noise
import math
import random
import sys

# params
resx = 256
res = (resx, resx)
turb_factor = 8
noctaves = 8
npeaks = 2
peakiness = 1.5
stratification = 0
base_height = 0.1

# random params
offset = (random.uniform(0, 1), random.uniform(0, 1))
peaks = []
for i in range(npeaks):
    peaks.append([random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)])

# maps
height_to_terrain = {
    0.0      : 'water',
    0.2      : 'beach',
    0.23     : 'forest',
    0.35     : 'mountain',
    0.45     : 'snow'
}

terrain_to_color = {
    'water'     : [0, 0, 255],
    'beach'     : [237, 201, 175],
    'forest'    : [34, 139, 34],
    'mountain'  : [100, 100, 30],
    'snow'      : [255, 255, 255]
}

print "Generating noise...\n"

noise_array = np.zeros(res, dtype=np.float32)
for x in range(res[0]):
    for y in range(res[1]):
        xc = (float(x) / res[0] * 2.0) - 1.0
        yc = (float(y) / res[1] * 2.0) - 1.0
        val = 0
        i = 1
        while i <= turb_factor:
            val += (noise.pnoise2((xc + offset[0]) * i, (yc + offset[1]) * i, octaves=noctaves)) / i
            i *= 2
        val = (val + 1.0) / 2.0
        noise_array[x, y] = val

        sys.stdout.write("\r{:3.2f}%".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

print "\ndone.\nGenerating terrain..."
terrain_array = np.zeros(res, dtype=np.float32)
for x in range(res[0]):
    for y in range(res[1]):
        val = noise_array[x, y]
        val = math.pow(val, peakiness)

        # make peaks
        xc = (float(x) / res[0] * 2.0) - 1.0
        yc = (float(y) / res[1] * 2.0) - 1.0
        dist = 1
        for peak in peaks:
            dist = min(dist, math.sqrt((xc - peak[0]) * (xc - peak[0]) + (yc - peak[1]) * (yc - peak[1])))
        val = val * (1.0 - 1.0 * math.pow(dist, 1.0))

        # increase by base height
        val += base_height
        terrain_array[x, y] = val

        sys.stdout.write("\r{:3.2f}% done ".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

print "\ndone.\nGenerating image..."
im_array = np.zeros((res[0], res[1], 3), dtype=np.uint8)
for x in range(res[0]):
    for y in range(res[1]):
        val = terrain_array[x, y]

        # discretize steps
        if stratification > 0:
            val = int(val * stratification) / float(stratification)

        # get color
        height_val = 0.0
        for height in height_to_terrain.keys():
            if val > height:
                height_val = max(height_val, height)
        color = terrain_to_color[height_to_terrain[height_val]]
        new_color = [0, 0, 0]
        val = min(max(val, 0.0), 1.0)
        new_color[0] = int(color[0] * 2.0 * val)
        new_color[1] = int(color[1] * 2.0 * val)
        new_color[2] = int(color[2] * 2.0 * val)
        im_array[x, y, :] = new_color

        sys.stdout.write("\r{:3.2f}% done ".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

im = Image.fromarray(im_array, mode="RGB")
Image._show(im)

print "\ndone.\nIslands generated with offset (" + str(offset[0]) + ", " + str(offset[1]) + ") and peaks:"
for peak in peaks:
    print peak
