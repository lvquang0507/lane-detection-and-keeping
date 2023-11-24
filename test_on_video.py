import cv2 as cv
import argparse
import lane_detect
import curve_lane

parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--lane",
    dest="lane_type",
    default="c",
    help="Lane type: [s]traight | [c]urve",
)

args = parser.parse_args()

vid = cv.VideoCapture("./videos/test1.mp4")

while vid.isOpened():
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if args.lane_type == "s":
        lane_detect.detect(frame)
    elif args.lane_type == "c":
        curve_lane.detect(frame)
    else:
        print("Invalid args")
    cv.waitKey(10)
    if cv.waitKey(1) == ord("q"):
        break
vid.release()
cv.destroyAllWindows()
