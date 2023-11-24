import cv2 as cv
import numpy as np


def detect(frame):
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    original_image = frame.copy()

    k_size = 5
    blur_img = cv.GaussianBlur(gray_img, ksize=(k_size, k_size), sigmaX=0.6, sigmaY=0.5)
    _, threshold_img = cv.threshold(blur_img, 125, 255, cv.THRESH_BINARY)
    # adp_img = cv.adaptiveThreshold(
    #     blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 2
    # )
    canny_img = cv.Canny(threshold_img, 120, 210)

    msk = roi(canny_img)
    lines = cv.HoughLinesP(
        msk, 2, np.pi / 180, 20, np.array([]), minLineLength=40, maxLineGap=20
    )

    lines_img, color_image = draw_line(gray_img, lines, color_img=original_image)

    if color_image is not None:
        cv.imshow("Color", color_image)
    else:
        cv.imshow("lines", lines_img)

    cv.imshow("Gray", gray_img)


def roi(image):
    mask = np.zeros_like(image)

    height, width, *_ = image.shape
    bottom_left = [width * 0, height * 0.9]
    bottom_right = [width * 1, height * 0.9]
    top_left = [width * 0.1, height * 0.25]
    top_right = [width * 0.9, height * 0.25]

    vtx = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    fill = cv.fillPoly(mask, vtx, 255)
    # cv.imshow("filled", fill)

    masked_img = cv.bitwise_and(image, mask)

    return masked_img


def draw_line(image, lines, color_img=None):
    line_image = np.zeros_like(image)
    height, width, *_ = line_image.shape

    right_slope = []
    left_slope = []
    right_intercept = []
    left_intercept = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope = (y1 - y2) / (x1 - x2)
            # print(slope)
            # print(f"x1={x1};x2={x2}")
            # cv.circle(image, (x1, y1), 5, [255, 0, 0], -1)
            # cv.circle(image, (x2, y2), 5, [0, 0, 255], -1)
            # cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            if np.abs(slope) > 0.8:
                if x1 > 180:
                    y_intercept = y2 - slope * x2
                    right_slope.append(slope)
                    right_intercept.append(y_intercept)
                elif x1 < 100:
                    y_intercept = y2 - slope * x2
                    left_slope.append(slope)
                    left_intercept.append(y_intercept)
            elif slope < 1.2:
                pass
            print(
                f"Right slope: {len(right_slope)}; Left slope: {len(left_slope)}",
            )

        try:
            # Calculate mean base on 30 prev values
            left_intercept_mean = np.mean(left_intercept[-30:])
            right_intercept_mean = np.mean(right_intercept[-30:])

            left_slope_mean = np.mean(left_slope[-30:])
            right_slope_mean = np.mean(right_slope[-30:])

            left_line_x1 = int(
                np.abs((0.8 * height - left_intercept_mean) / left_slope_mean)
            )
            left_line_x2 = int(
                np.abs((0.1 * height - left_intercept_mean) / left_slope_mean)
            )

            right_line_x1 = int(
                (0.8 * height - right_intercept_mean) / right_slope_mean
            )
            right_line_x2 = int(
                (0.1 * height - right_intercept_mean) / right_slope_mean
            )

            pts = np.array(
                [
                    [left_line_x1, int(0.8 * height)],
                    [left_line_x2, int(0.1 * height)],
                    [right_line_x2, int(0.1 * height)],
                    [right_line_x1, int(0.8 * height)],
                ],
                np.int32,
            )

            pts = pts.reshape((-1, 1, 2))

            if color_img is not None:
                cv.fillPoly(color_img, pts=[pts], color=(200, 0, 0))
            else:
                cv.fillPoly(line_image, pts=[pts], color=(255, 0, 0))
        except ValueError:
            pass
    else:
        print("mt")

    return line_image, color_img
