import cv2
import matplotlib.pyplot as plt
import numpy as np

image_name = 'frame_2'
image = cv2.imread(f'./{image_name}.jpg', 0)

x_size = int(image.shape[0])
y_size = int(image.shape[1])  # (936, 936)

softbody_index = {}

for row in range(x_size):
    for column in range(y_size):
        if image[row][column] > 200:
            if row not in softbody_index:
                softbody_index[row] = [column]
            else:
                softbody_index[row].append(column)


rows = len(softbody_index.keys())

boundary_index = []

for row in softbody_index.keys():
    items = softbody_index[row]
    if len(items) == 1:
        boundary_index.append((row, items[0]))
    else:
        boundary_index.append((row, items[0]))
        boundary_index.append((row, items[-1]))

print(len(boundary_index))
edge_image = np.zeros((x_size, y_size))
for pixel_index in boundary_index:
    edge_image[pixel_index[0]][pixel_index[1]] = 255

cv2.imshow(f'boundary image', edge_image)
cv2.imwrite(f'{image_name}_boundary.jpg', edge_image)

plt.figure(figsize=(3, 3), dpi=300)
plt.axis([0, 1, 0, 1])
xs = [pixel_index[1] / y_size for pixel_index in boundary_index]
ys = [(x_size - pixel_index[0]) / x_size for pixel_index in boundary_index]


plt.scatter(xs, ys, s=0.05)
plt.show()
plt.savefig(f'{image_name}_point_cloud.png')
