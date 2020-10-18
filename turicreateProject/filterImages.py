import cv2
import os
import numpy as np
from apply_filter import applyFilter as filter

edgeFilter = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

customFilter = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])


def loadImagesFromFolder(folder):
    """
    This function takes in a path to a directory that conains more some data.
    In the directory there is folders and each folder has images. This function 
    loops through all the directories and gets the images and applies a filter
    to the images which are then saved to another folder.
    """
    for subFolder in os.listdir(folder):
        if subFolder[0] == ".":
            pass
        else:
            for filename in os.listdir(folder + "/" + subFolder):
                if filename[0] == ".":
                    pass
                else:
                    # 
                    directory = "./expandedDataset/" + subFolder
                    # set image path
                    image_path = folder + "/" + subFolder + "/" + filename
                    # use opencv to read the image
                    img = cv2.imread(image_path) 

                    print(image_path)
                    # apply filter
                    # filteredImg = filter(img, edgeFilter)

                    # black and white image
                    blackAndGrey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                    # make a slighlty blurred image
                    blurred = cv2.GaussianBlur(img.copy(), (21,21), 0)
                    # make a saturated image
                    saturated = cv2.filter2D(img.copy(), -100, customFilter)
                    # flip image horizontally
                    horizontal = cv2.flip(img.copy(), 1)
                    # flip image vertically
                    veritcally = cv2.flip(img.copy(), 0)

                    # save it to new folder
                    cv2.imwrite(os.path.join(directory , filename), img.copy())
                    cv2.waitKey(0)

                    cv2.imwrite(os.path.join(directory , filename), blackAndGrey)
                    cv2.waitKey(0)
                   
                    cv2.imwrite(os.path.join(directory , filename), blurred)
                    cv2.waitKey(0)

                    cv2.imwrite(os.path.join(directory , filename), saturated)
                    cv2.waitKey(0)

                    cv2.imwrite(os.path.join(directory , filename), horizontal)
                    cv2.waitKey(0)

                    cv2.imwrite(os.path.join(directory , filename), veritcally)
                    cv2.waitKey(0)


    return 0

print(loadImagesFromFolder("./dataset"))
