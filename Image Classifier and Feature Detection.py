import cv2
import numpy as np
import os

path ='F:\My Projects\OpenCV\ImagesPath'
images=[]
classNames =[]
mylist=os.listdir(path)
print ('Total classes detected is', len(mylist))
for i in mylist:
    imgCur=cv2.imread(f'{path}/{i}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(i)[0])

print(classNames)
orb= cv2.ORB_create(nfeatures=1000)
def findDes(images):
    deslist=[]
    for img in images:
        img= cv2.resize(img, (500, 700))
        kp, des=orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist


def findID(img,deslist, thres=15):
    kp2,des2= orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal=-1
    try:

         for des in deslist:
             matches= bf.knnMatch(des,des2, k=2)
             good = []
             for m, n in matches:
                 if m.distance < 0.75 * n.distance:
                     good.append([m])
             matchList.append(len(good))

    except:
        pass
    #print(matchList)
    if len(matchList)!=0:
        if max(matchList)> thres:
            finalVal=matchList.index(max(matchList))
    return finalVal








deslist=findDes(images)
print(len(deslist))



cap=cv2.VideoCapture(1)

while True:
    success, img2 =cap.read()
    imageOriginal= img2.copy()
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    id= findID(img2,deslist)
    if id!= -1:
        cv2.putText(imageOriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



    cv2.imshow("img2",imageOriginal)
    cv2.waitKey(1)












# kp1, des1= orb.detectAndCompute(img1,None)
# kp2, des2= orb.detectAndCompute(img2, None)
# #print(des1[499])
#
# imgKp1= cv2.drawKeypoints(img1, kp1,None)
# imgKp2= cv2.drawKeypoints(img2, kp2,None)
#


#
# img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#
#
# # cv2.imshow("Kp1", imgKp1)
# # cv2.imshow("Kp2", imgKp2)
# cv2.imshow("Matches", img3)

