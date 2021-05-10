import cv2

src = cv2.imread(f'./walls.jpg')
dst = src[:300, :300]
cv2.imwrite(f'walls.jpg', dst)
