import cv2
import numpy as np
from scipy.optimize import minimize

from app.utils import gradient_magnitude, distance, curvature

image = cv2.imread("data/dark_circle.png")

if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
elif len(image.shape) == 4:
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
else:
    gray = image.clone()

height, width = gray.shape


# start points
start_points = [
    (250, 150),
    (250, 200),
    (250, 250),
    (250, 300),
    (250, 350),
]

start_points_length = len(start_points)

flatten_start_points = []
for x, y in start_points:
    flatten_start_points.append(x)
    flatten_start_points.append(y)

start_point = np.array(flatten_start_points)

# gradient magnitude
mag = 255 - gradient_magnitude(gray, kernel_size=32)*100

color_mag = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

def obj_func(p):
    # draw
    global color_mag

    temp = color_mag.copy()

    for i in range(0, len(p), 2):
        x = int(p[i])
        y = int(p[i + 1])
        temp = cv2.circle(temp, (x, y), 5, (255, 0, 0), 5)

    # display results
    cv2.imshow('frame', temp)
    cv2.waitKey(10)

    # continuity
    sum_distance = 0

    for i in range(0, len(p)-2, 2):
        sum_distance += distance(p[i], p[i+1], p[i+2], p[i+3])

    mean_distance = 2 * sum_distance / len(p)

    continuity_acc = (mean_distance - sum_distance)**2

    # curvature
    curvature_acc = 0

    for i in range(2, len(p)-2, 2):
        curvature_acc += curvature(p[i-2], p[i-1], p[i], p[i+1], p[i+2], p[i+3])

    # magintude
    mag_acc = 0

    for i in range(0, len(p), 2):
        x0 = int(p[i])
        y0 = int(p[i+1])

        value = mag[y0, x0]

        print(value)

        mag_acc += value

    print(mag_acc)

    # return value0 + value1 + value2 + abs(10 - dist0)*0.1 + abs(10 - dist1)*0.1
    return mag_acc # continuity_acc + curvature_acc + mag_acc

res = minimize(obj_func, start_point, options={'eps': 3})
print(res.x)

print(start_points)

# print results
# for x, y in start_points:
#     color_mag = cv2.circle(color_mag, (x, y), 5, (255, 255, 0), 5)

color_mag = color_mag.copy()

for i in range(0, len(res.x), 2):
    x = int(res.x[i])
    y = int(res.x[i+1])
    color_mag = cv2.circle(color_mag, (x, y), 5, (255, 0, 0), 5)

# display results
cv2.imshow('frame', color_mag)
cv2.waitKey(10000)

# clean results
cv2.destroyAllWindows()
