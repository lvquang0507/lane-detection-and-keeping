import cv2 as cv
from lane_detect import lane_detect

vid = cv.VideoCapture("./videos/test1.mp4")

while vid.isOpened():
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    lane_detect(frame)
    cv.waitKey(10)
    if cv.waitKey(1) == ord("q"):
        break
vid.release()
cv.destroyAllWindows()
