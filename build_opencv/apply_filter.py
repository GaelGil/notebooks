import numpy as np
import cv2

edgeFilter = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])



img = cv2.imread("sneaker.png")


def applyFilter(img, filter):
    """
    This function takes in a image as its argument (numpy array)
    and a filter (some array with n dimmensions). We will multiply
    the filter by the array to get a filtered image.
    """

    # apply grey filter to make it easier
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    cols = gray.shape[1]
    for x in range(0, rows-3):
        for y in range(0, cols-3):  
            gray[x][y] = filter[0][0]*gray[x][y]
            gray[x][y+1] = filter[0][1]*gray[x][y+1]
            gray[x][y+2] = filter[0][2]*gray[x][y+2]
            gray[x+1][y] = filter[1][0]*gray[x+1][y]
            gray[x+1][y+1] = filter[1][1]*gray[x+1][y+1]
            gray[x+1][y+2] = filter[1][2]*gray[x+2][y+2]
            gray[x+2][y] = filter[2][0]*gray[x+2][y]
            gray[x+2][y+1] = filter[2][1]*gray[x+2][y+1]
            gray[x+2][y+2] = filter[2][2]*gray[x+2][y+2]

    
    return gray

