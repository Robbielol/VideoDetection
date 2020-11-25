import cv2, time, pandas
from datetime import datetime

firstFrame=None
statusList=[None,None]
times=[]
dataframe=pandas.DataFrame(columns=["Start", "End"])
video=cv2.VideoCapture(0)

while True:
    check,frame = video.read()
    status=0
    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grayFrame=cv2.GaussianBlur(grayFrame,(21,21),0)

    if firstFrame is None:
        firstFrame = grayFrame
        continue


    deltaFrame=cv2.absdiff(firstFrame,grayFrame)
    thresFrame=cv2.threshold(deltaFrame, 30, 255,cv2.THRESH_BINARY)[1]
    thresFrame=cv2.dilate(thresFrame, None, iterations=2)

    (cont,_) = cv2.findContours(thresFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for counter in cont:
        if cv2.contourArea(counter) < 10000:
            continue
        status=1
        (x, y, z, w)= cv2.boundingRect(counter)
        cv2.rectangle(frame, (x,y),(x+z, y+w), (0,255,0),4)

    statusList.append(status)
    if statusList[-1]==1 and statusList[-2] == 0:
        times.append(datetime.now())
    if statusList[-1]==0 and statusList[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Video", frame)
    key=cv2.waitKey(1)

    if key==ord("q"):
        break

for i in range(0, len(times),2):
    df=dataframe.append({"Start":times[i],"End": times[i+1]}, ignore_index=True)

dataframe.to_csv("MotionCaptureTimes.csv")
video.release()
cv2.destroyAllWindows()