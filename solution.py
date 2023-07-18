import glob
import cv2
from imaging_interview import preprocess_image_change_detection, compare_frames_change_detection
import numpy as np
import os

def data_cleanup(path, threshold):

    # import all the images from the path
    img_list = glob.glob(path+'/*.png')

    #downsize all the images to same size to be able to compare them
    for img in img_list:
        try:
            image = cv2.imread(img)
            image = cv2.resize(image, (640, 480))
            cv2.imwrite(img, image)
        except:
            os.remove(img) # remove corrupted images, that cannot be opened by cv2

    try:
        img1_index = 0
        deleted = 0
        img_list = sorted(glob.glob(path+'/*.png'))
        
        for i in range(len(img_list)):
            
            img2_index = i + 1

            img1 = cv2.imread(img_list[img1_index])
            img2 = cv2.imread(img_list[img2_index])

            gray1 = preprocess_image_change_detection(img1)
            gray2 = preprocess_image_change_detection(img2)

            _, _, thresh = compare_frames_change_detection(gray1, gray2, 1e4)

            uniques, counts = np.unique(thresh, return_counts=True)

            if len(uniques) == 1:
                os.remove(img_list[img2_index])
                print("Deleted " + img_list[img2_index])
                deleted += 1
                continue
            else:
                percent_diff = (1-counts[0]/(counts[0]+counts[1]))*100 # computing percenatge of difference between the images, if difference < input parameter, delete the image
                if percent_diff < threshold:                           
                    os.remove(img_list[img2_index])
                    print("Deleted " + img_list[img2_index])
                    deleted += 1
                    continue
                else:                                                  # else use the new image as reference for future images 
                    img1_index = img2_index
    except IndexError:
        print('Done! Deleted ' + str(deleted) + ' files.')

def main():

    path = '/home/rushabh/Interview/Kopernikus/dataset-candidates-ml_2/dataset'
    threshold = 10

    data_cleanup(path, threshold)

if __name__ == "__main__":

    main()
