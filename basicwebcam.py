import numpy as np
import cv2

# get camera
cap = cv2.VideoCapture(0)

printed = 0
# eye cascade rule (to find eyes)
lCascade = cv2.CascadeClassifier('/Users/floodric/18/haarcascades/haarcascade_lefteye_2splits.xml')
rCascade = cv2.CascadeClassifier('/Users/floodric/18/haarcascades/haarcascade_righteye_2splits.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #idk how this works but it finds the eyes
    leye = lCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=13,
      minSize=(30, 30),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    reye = rCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=13,
      minSize=(30, 30),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # try to get the correct eye
    # reye[2] is width
    if(len(leye) > 0 and len(reye) > 0):
      av = [((leye[0][0] + (reye[0][0]+reye[0][2]) )/ 2),((leye[0][1] + reye[0][1]+reye[0][3]) / 2)]
    
      cv2.circle(gray, (av[0],av[1]), 10, (255,0,0))

    # draw boxes over the eyes
    for (x, y, w, h) in leye:
      cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in reye:
      cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
