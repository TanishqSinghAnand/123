import cv2
import time
########################

pTime = 0


########################


cap = cv2.VideoCapture(0)

while True:
 success, img = cap.read()

 cTime = time.time()
 fps = 1/(cTime-pTime)
 pTime = cTime
 cv2.putText(img, f'FPS : {int(fps)}', (10, 50),cv2.FONT_HERSHEY_PLAIN, 3, (150, 255, 75), 3)

 cv2.imshow('Image', img)
 if cv2.waitKey(1) & 0xFF == ord('q'):
   break

cap.release()
cv2.destroyAllWindows()