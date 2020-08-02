import cv2
import numpy as np

print("OpenCV version:")
print(cv2.__version__)

# regular img
img = cv2.imread("sneaker.png")


# 
im = cv2.imread("sneaker.png")
im = np.asarray(im)
print(im)
print(im.shape)



# gray color img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blury img
img_blur = cv2.GaussianBlur(img, (7,7), 0)

# cool filter img
img_canny = cv2.Canny(img, 100, 100)


# https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#line
# horizontal green line
cv2.line(img, pt1=(0, 150), pt2=(600, 150), color=(0, 255, 0), thickness=3)

# vertical green line
cv2.line(img, (280, 0), (280, 400), (0, 255, 0), thickness=3)




b = img.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


g = img.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = img.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0


upper_left_corner = img.copy()
# set blue and green channels to 0
# upper_left_corner[100, 100, :] = 0
# upper_left_corner[:, :, 1] = 0

# middle of img
u =  upper_left_corner[100:200, 100:200, :]


list = [1, 2, 3, 4, 5, 6, 7]


# cv2.imshow('upper lefct', u)

# RGB - Blue
# cv2.imshow('B-RGB', b)

# # RGB - Green
# cv2.imshow('G-RGB', g)

# # RGB - Red
# cv2.imshow('R-RGB', r)


cv2.imshow("Over the Clouds", img)
cv2.imshow("Over the Clouds - gray", gray)
# cv2.imshow("Over the Clouds - blur", img_blur)
# cv2.imshow("Over the Clouds - canny", img_canny)
# cv2.imshow("Over the Clouds - upperleft", upper_left_corner)





cv2.waitKey(0)
# cv2.destroyAllWindows()



