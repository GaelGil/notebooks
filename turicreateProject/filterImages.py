import cv2
import os
import numpy as np
from apply_filter import applyFilter as filter

edgeFilter = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])



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
                    directory = "./newDataset/" + subFolder
                    # set image path
                    image_path = folder + "/" + subFolder + "/" + filename
                    # use opencv to read the image
                    img = cv2.imread(image_path) 
                    # apply filter
                    filteredImg = filter(img, edgeFilter)
                    # save it to new folder
                    cv2.imwrite(os.path.join(directory , filename), filteredImg)
                    cv2.waitKey(0)


    return 0

print(loadImagesFromFolder("./dataset"))