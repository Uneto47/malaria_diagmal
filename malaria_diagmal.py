from skimage.data import coins
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import hough_circle_peaks, hough_circle
from skimage.filters import sobel
import numpy as np
from skimage.morphology import binary_opening, binary_closing
from skimage import color
from skimage.draw import circle_perimeter

image = imread('DSCN0267_JPG.rf.badebfd7d1ef62baa0dd43e0e1ccb80f.jpg')
image = color.rgb2gray (image)

plt.imshow(image,cmap='gray')

edges = sobel (image)
plt.imshow(edges,cmap='gray')

h = np.histogram (edges.ravel(),bins=256)

plt.plot (h[0],'-k')

binary = edges.copy()
limiar = edges.max() * 0.05

binary[binary <= limiar] = 0
binary[binary >  limiar] = 1

plt.imshow(binary, cmap='gray')

binary = binary_closing (binary)
binary = binary_opening (binary)
plt.imshow (binary,cmap='gray')

raios = [75]
hough_grade = hough_circle (binary, raios)

grade = hough_grade[0]
plt.imshow (grade,cmap='gray')

acumuladores, a, b, raio = hough_circle_peaks (hough_grade, raios, total_num_peaks=3)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))

image_color = color.gray2rgb(image)

for center_y, center_x, radius in zip(b, a, raio):

    # contorno mais escuro (ligeiramente maior)
    cy1, cx1 = circle_perimeter(
        center_y, center_x, radius + 1, shape=image.shape
    )
    image_color[cy1, cx1] = (0, 120, 0)   # verde escuro

    # círculo chamativo (ligeiramente menor)
    cy2, cx2 = circle_perimeter(
        center_y, center_x, radius, shape=image.shape
    )
    image_color[cy2, cx2] = (0, 255, 80)  # verde neon

ax.imshow(image_color)
ax.axis("off")

