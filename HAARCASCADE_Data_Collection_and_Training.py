import cv2
import os
import time

path= 'F:\My Projects\OpenCV\Codes\Collected Data (HAARCASCADE)\Data_Collected'
camBright= 180
moduleval= 10
minBlur=500
grayImg=False
saveData= True
showImg=True
imgW=180
imgH=120

global countFolder
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,camBright)

count = 0
countSave =0

def saveDataFunc():
    global countFolder
    countFolder=0
    while os.path.exists(path+str(countFolder)):
        countFolder+=1
    os.mkdir(path+str(countFolder))

if saveData:saveDataFunc()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (imgW, imgH))
    if grayImg:img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if saveData:
        blur=cv2.Laplacian(img,cv2.CV_64F).var()
        if count % moduleval ==0 and blur>minBlur:
            nowTime= time.time()
            cv2.imwrite(path + str(countFolder) +
                        '/' + str(countSave) + "_" + str(int(blur)) + "_" + str(nowTime) + ".png", img)
            countSave+=1
        count+=1
    if showImg:
                cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





