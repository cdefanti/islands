import numpy as np
from PIL import Image
import noise
import math
import random
import sys

# params
res = (256, 256, 3)
narray = np.zeros(res, dtype=np.int8)
turb_factor = 4
noctaves = 4
npeaks = 2
peakiness = 1.2
stratification = 0

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
    0.47     : 'snow'
}
terrain_to_color = {
    'water'     : [0, 0, 255],
    'beach'     : [237, 201, 175],
    'forest'    : [34, 139, 34],
    'mountain'  : [100, 100, 30],
    'snow'      : [255, 255, 255]
}

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
        val = math.pow(val, peakiness)

        # penalize near border
        dist = 1
        for peak in peaks:
            dist = min(dist, math.sqrt((xc - peak[0]) * (xc - peak[0]) + (yc - peak[1]) * (yc - peak[1])))
        val = val * (1.0 - 1.0 * math.pow(dist, 1.0))
        if stratification > 0:
            val = int(val * stratification) / float(stratification)

        # get color
        height_val = 0.0
        for height in height_to_terrain.keys():
            if val > height:
                height_val = max(height_val, height)
        color = terrain_to_color[height_to_terrain[height_val]]
        new_color = [0, 0, 0]
        new_color[0] = (color[0] * np.clip(2.0 * val, 0, 1))
        new_color[1] = (color[1] * np.clip(2.0 * val, 0, 1))
        new_color[2] = (color[2] * np.clip(2.0 * val, 0, 1))
        narray[x, y, :] = new_color
    sys.stdout.write("\r{:3.2f}% done ".format(100 * float(x) / (res[0])))
    sys.stdout.flush()

im = Image.fromarray(narray, mode="RGB")
Image._show(im)
