# Kopernikus_data_cleanup_task

## Question and Answers

1. What did you learn after looking on our dataset?
##
The dataset consists of set of images from a parking garage taken from different cameras at different time of the day. The pictures seem to be taken at a resolution of 1080p (1920x1080), although the pictures from the camera c10 have been downsized to 640x480 (except for one). There is a mismatch of resolution among the images. This makes some form of pre processing necessary to be able to work with this dataset.

4. How does you program work?\ We firstly start with some data preparation - ensuring that all images have same size and deleting some corrupted file. Inside a for loop, we start with the 1st image as a refernce image and compare the next image wth this reference image using the given function. The output 'thresh' which contains the difference between the two images in the form of a map with values 255 and 0 is used. We compute a percentage of difference between the two images and if the percentage is less than a threshold that can be provided as an argument, we delete this image and move onto the next one with the same refernce image. IF there is a significant difference bewteen the images, we keep the image and change the reference image to this new file and continue the process.

5. What values did you decide to use for input parameters and how did you find these values?\ Minimum contour area was one of the parameters that was used in the compare_frames_change_detection function. I set it to value of 1e4, as I thought a contours size of 100x100 pixels would make sense to measure the changes in the image of size 640x480. One more parameter was added in the program that can be passed as an argument, and that is the threshold for percentage difference between the images. This value is provided as a percentage, for example, if this value is set as 20, then images that have a percent_diff score less than 20 percent will be deleted.

6. What you would suggest to implement to improve data collection of unique cases in future?\ One improvement that I can think of is to program the data collection pipelines in a manner that we don't capture the data at 30 frames per second, or if we do then intentionally drop some frames, because these consecutive frames would look mostly identical and won't have much learning potential for the network. Getting rid of these frames during data genration would make the entire data processing pipline more efficient.
