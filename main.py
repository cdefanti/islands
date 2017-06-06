import numpy as np
from PIL import Image
import noise
import math
import random
import sys
import scipy.signal

# params
resx = 256
res = (resx, resx)
turb_factor = 4
noctaves = 4
npeaks = 3
peakiness = 0.8
stratification = 0
base_height = -0.10

simulate_rainfall = False

# random params
base_temperature = random.uniform(0, 1)
base_humidity = random.uniform(0, 1)
terrain_c = (random.uniform(0, 1), random.uniform(0, 1))
temperature_c = (random.uniform(0, 1), random.uniform(0, 1))
humidity_c = (random.uniform(0, 1), random.uniform(0, 1))
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
    'snow'      : [255, 255, 255],
    'lake'      : [255, 0, 0]
}

print "Generating noise...\n"

# terrain
noise_array = np.zeros(res, dtype=np.float32)
for x in range(res[0]):
    for y in range(res[1]):
        xc = (float(x) / res[0] * 2.0) - 1.0
        yc = (float(y) / res[1] * 2.0) - 1.0

        val = 0
        i = 1
        while i <= turb_factor:
            val += (noise.pnoise2((xc + terrain_c[0]) * i, (yc + terrain_c[1]) * i, octaves=noctaves)) / i
            i *= 2
        val = (val + 1.0) / 2.0
        noise_array[x, y] = val

        sys.stdout.write("\r{:3.2f}%".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

# temperature
temperature_array = np.zeros(res, dtype=np.float32)
for x in range(res[0]):
    for y in range(res[1]):
        xc = (float(x) / res[0] * 2.0) - 1.0
        yc = (float(y) / res[1] * 2.0) - 1.0

        val = noise.pnoise2((xc + temperature_c[0]) * i, (yc + temperature_c[1]) * i, octaves=noctaves)
        val = (val + 1.0) / 2.0
        temperature_array[x, y] = val

        sys.stdout.write("\r{:3.2f}%".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

# humidity
humidity_array = np.zeros(res, dtype=np.float32)
for x in range(res[0]):
    for y in range(res[1]):
        xc = (float(x) / res[0] * 2.0) - 1.0
        yc = (float(y) / res[1] * 2.0) - 1.0

        val = noise.pnoise2((xc + humidity_c[0]) * i, (yc + humidity_c[1]) * i, octaves=noctaves)
        val = (val + 1.0) / 2.0
        humidity_array[x, y] = val

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

print "\ndone.\nGenerating gradient..."
xFilter = [[-1, 0, 1]]
yFilter = [[-1], [0], [1]]
gradX = scipy.signal.convolve2d(terrain_array, xFilter, mode='same')
gradY = scipy.signal.convolve2d(terrain_array, yFilter, mode='same')
grad_array = np.zeros(res, dtype=np.float32)
graddir_array = np.zeros((res[0], res[1], 2), dtype=np.float32)
gradim_array = np.zeros((res[0], res[1], 3), dtype=np.uint8)
for x in range(res[0]):
    for y in range(res[1]):
        grad_array[x, y] = math.sqrt(gradX[x, y] * gradX[x, y] + gradY[x, y] * gradY[x, y])
        graddir_array[x, y] = (gradX[x, y], gradY[x, y])
        gradim_array[x, y] = int(5 * 255 * grad_array[x, y])

        sys.stdout.write("\r{:3.2f}%".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

im = Image.fromarray(gradim_array, mode="RGB")
Image._show(im)

if simulate_rainfall:
    print "\ndone.\nSimulating rainfall..."
    MAXITER = 100
    MINDELTA = 0.001
    water_array = np.zeros(res, dtype=np.float32)
    for x in range(res[0]):
        for y in range(res[1]):
            if terrain_array[x, y] < 0.2:
                continue
            coord = [float(x) / res[0] * 2.0 - 1.0, float(y) / res[1] * 2.0 - 1.0]
            for j in range(MAXITER):
                pos = [int((coord[0] + 1.0) * (res[0] / 2)), int((coord[1] + 1.0) * (res[1] / 2))]
                if pos[0] in range(res[0]) and pos[1] in range(res[1]):
                    water_array[pos[0], pos[1]] += 1.0 / (res[0] * res[1] * MAXITER)
                    delta = graddir_array[pos[0], pos[1]]
                else:
                    break
                if delta[0] * delta[0] + delta[1] * delta[1] > MINDELTA * MINDELTA:
                    coord += delta
                else:
                    water_array[pos[0], pos[1]] += float(MAXITER - j) / (res[0] * res[1] * MAXITER)
                    break

            sys.stdout.write("\r{:3.2f}%".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
            sys.stdout.flush()
    print np.max(water_array)
    water_image = np.zeros((res[0], res[1], 3), dtype=np.uint8)
    for x in range(res[0]):
        for y in range(res[1]):
            val = int(water_array[x, y] / np.max(water_array) * 255)
            water_image[x, y] = [val, val, val]

    im = Image.fromarray(water_image, mode="RGB")
    Image._show(im)

print "\ndone.\nGenerating image..."
im_array = np.zeros((res[0], res[1], 3), dtype=np.uint8)
for x in range(res[0]):
    for y in range(res[1]):
        val = terrain_array[x, y]

        # discretize steps
        if stratification > 0:
            val = int(val * stratification) / float(stratification)

        if simulate_rainfall and water_array[x, y] > 0.0001:
            color = terrain_to_color['water']
        else:
            # get color
            height_val = 0.0
            for height in height_to_terrain.keys():
                if val > height:
                    height_val = max(height_val, height)
            color = terrain_to_color[height_to_terrain[height_val]]

        new_color = [0, 0, 0]
        val = min(max(val, 0.0), 0.5)
        new_color[0] = int(color[0] * 2.0 * val)
        new_color[1] = int(color[1] * 2.0 * val)
        new_color[2] = int(color[2] * 2.0 * val)
        im_array[x, y, :] = new_color

        sys.stdout.write("\r{:3.2f}% done ".format(100 * float(x * res[1] + y) / (res[0] * res[1])))
        sys.stdout.flush()

im = Image.fromarray(im_array, mode="RGB")
Image._show(im)

offset_matrix = np.zeros((3, 2), dtype=float)
offset_matrix[0, :] = terrain_c
offset_matrix[1, :] = temperature_c
offset_matrix[2, :] = humidity_c
print "\ndone.\nIslands generated with offsets:"
print offset_matrix
print "and peaks:\n"
for peak in peaks:
    print peak
