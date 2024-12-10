from scipy.io import savemat, loadmat
import cv2  #install opencv-python and opencv-contrib-python
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import sys

class matching:
    def __init__(self,ImageDirectory,ImageFormat) -> None:
        self.ImgDir = ImageDirectory
        self.ImageFormat = ImageFormat

        self.DescriptorType = ''

        self.ImageNames = {}
        count = 0
        for filename in os.listdir(self.ImgDir):
            #Check if file is of the correct format
            if not filename.endswith(self.ImageFormat): 
                continue
            self.ImageNames[count] = filename
            count+=1

        self.NImages = count

        #Keypoints
        self.kp: List[List[cv2.KeyPoint]] = [[] for _ in range(self.NImages)]
        #Descriptors
        self.des: List[Any] = [None for _ in range(self.NImages)]
        self.mask: Dict[Tuple[int, int], List[bool]] = {}

        self.Results = {}

    def __ReinitializeVar__(self):

        self.ImageNames = {}
        count = 0
        for filename in os.listdir(self.ImgDir):
            #Check if file is of the correct format
            if not filename.endswith(self.ImageFormat): 
                continue
            self.ImageNames[count] = filename
            count+=1

        self.NImages = count

        #Keypoints
        self.kp: List[List[cv2.KeyPoint]] = [[] for _ in range(self.NImages)]
        #Descriptors
        self.des: List[Any] = [None for _ in range(self.NImages)]
        self.mask: Dict[Tuple[int, int], List[bool]] = {}

    def __CheckForUpdates__(self,NewDescriptorType):

        if NewDescriptorType != self.DescriptorType:
            self.DescriptorType = NewDescriptorType
            return True

        count = 0
        for filename in os.listdir(self.ImgDir):
            #Check if file is of the correct format
            if not filename.endswith(self.ImageFormat): 
                continue

            if not filename in self.ImageNames.values():
                return True   
              
            count +=1   
            
        if count != self.NImages:
            return True

        return False
    
    def __GetKPAndDesc__(self, index):
        desc1 = []
        pt1 = []
        for i, kp1 in enumerate(self.kp[index]):

            # Get the matching keypoints coordinates
            pt1.append(kp1.pt)  # Point from the first image
            
            # Get the corresponding descriptors
            desc1.append(self.des[index][i])

        pt1 = np.array(pt1)
        desc1 = np.array(desc1)

        return pt1, desc1

    def __RunDetector__(self,detector):
            for index1, filename1 in self.ImageNames.items():
                image1 = cv2.imread(self.ImgDir+filename1) 
                img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                self.kp[index1],self.des[index1] = detector(img1)

    def __SiftDetec__(self,img1):
        #Create Descriptor function object
        DescFunc = cv2.SIFT_create()
        #Feature Detection
        kp1, des1 = DescFunc.detectAndCompute(img1,None)
        return kp1, des1
        
    def __OrbDetec__(self,img1):
        #Create Descriptor function object
        DescFunc = cv2.ORB_create()
        #Feature Detection
        kp1, des1 = DescFunc.detectAndCompute(img1,None)
        return kp1, des1
                
    def SIFT(self):
        if self.__CheckForUpdates__("SIFT_Seq"):
            self.__ReinitializeVar__()
            self.__RunDetector__(self.__SiftDetec__)
            self.Results = self.SaveToDictionary()
    
    def Orb(self):
        if self.__CheckForUpdates__("Orb_All"):
            self.__ReinitializeVar__()
            self.__RunDetector__(self.__OrbDetec__)
            self.Results = self.SaveToDictionary()

    def SaveToDictionary(self):
        Dict = {}

        for index in range(self.NImages):
            pt1, desc1 = self.__GetKPAndDesc__(index)

            filename1, _ = os.path.splitext(self.ImageNames[index])
            Dict['Feature_'+str(filename1)] = {'kp':pt1,'desc':desc1}
        return Dict
    
def saveMat(dict):
    #save mat file and open it as binary
    savemat("/tmp/data.mat",dict,long_field_names=True, do_compression=True)

def OpenMatFile(Directory):
    for filename in os.listdir(Directory):
        #Check if file is of the correct format
        if not filename.endswith(".mat"): 
            continue
        else:
            Dict = loadmat(Directory+filename)
            return Dict
    raise ValueError("No .mat file found in the Directory")

#Main for test purposes      
if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("This program needs one input argument.\nThis argument should be the file type of the images to search in the TestImages directory.\nEx.: python feature.py .jpg")
    else:
        FileType = sys.argv[1]
        test = matching("./TestImages/",FileType)
        test.SIFT()
        FeatureDict = test.SaveToDictionary()

        savemat("./kp.mat",FeatureDict,long_field_names=True, do_compression=True)
    