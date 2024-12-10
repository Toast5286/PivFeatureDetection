# PivFeatureDetection
 A tool to help obtain and store features of images into a .mat file

## Deploy

Make sure you have python and the following libraries installed:
- scipy
- numpy
- typing
- cv2

Create a "TestImages" directory in the same directory as the python program. Afterwards just add the images you want to extract the features in to the "TestImages" directory and run the following command:

```
  python feature.py .[image extention]
```
After the program runs, the .mat file with the data will be in the same directory as the python file.It will contain "Feature_[image name without the extention]" structure that contains the key points of that image and there descriptors.

# Example

```
  python feature.py .jpg
```