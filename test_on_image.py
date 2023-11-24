import cv2 as cv
import numpy as np


def detect():
    global img
    img = cv.imread("./imgs/frame0.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    _, bin_img = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY)

    warp = perspective_warp(
        image=bin_img,
        src=np.float32([(0.01, 0.83), (0.98, 0.78), (0.24, 0.4), (0.75, 0.4)]),
        dst=np.float32(
            [(0, 1 - 0.05), (1 - 0.05, 1 - 0.05), (0 + 0.05, 0 + 0.05), (1 - 0.05, 0)]
        ),
    )

    canny = cv.Canny(warp, 120, 210)
    cv.imshow("Warp", warp)
    cv.imshow("Original", img)
    cv.imshow("Canny", canny)
    cv.waitKey(0)


def roi(image, region):
    pass


def perspective_warp(image, src, dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    height, width, *_ = image.shape

    img_size = np.float32([width, height])

    src = src * img_size
    dst = dst * img_size

    print(src)
    for i, pt in enumerate(src):
        pt_int = np.int16(pt)
        print(pt_int)
        cv.circle(img, pt_int, 5, (255, 0, 0), -1)
        cv.putText(img, f"{i}", pt_int, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    transform_mat = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(
        image, transform_mat, (width, height), flags=cv.INTER_LANCZOS4
    )
    return warped


def sliding_window_search(
    img, windows_number=9, margin=150, min_pix=1, draw_windows=True
):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / windows_number)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idx = []
    right_lane_idx = []

    # Step through the windows one by one
    for window in range(windows_number):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(
                out_img,
                (win_x_left_low, win_y_low),
                (win_x_left_high, win_y_high),
                (100, 255, 255),
                3,
            )
            cv.rectangle(
                out_img,
                (win_x_right_low, win_y_low),
                (win_x_right_high, win_y_high),
                (100, 255, 255),
                3,
            )
        # Identify the nonzero pixels in x and y within the window
        good_left_idx = (
            (nonzero_y >= win_y_low)
            & (nonzero_y < win_y_high)
            & (nonzero_x >= win_x_left_low)
            & (nonzero_x < win_x_left_high)
        ).nonzero()[0]
        good_right_idx = (
            (nonzero_y >= win_y_low)
            & (nonzero_y < win_y_high)
            & (nonzero_x >= win_x_right_low)
            & (nonzero_x < win_x_right_high)
        ).nonzero()[0]
        # Append these indices to the lists
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)
        # If you found > min_pix pixels, recenter next window on their mean position
        if len(good_left_idx) > min_pix:
            left_x_current = np.int(np.mean(nonzero_x[good_left_idx]))
        if len(good_right_idx) > min_pix:
            right_x_current = np.int(np.mean(nonzero_x[good_right_idx]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty**2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty**2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


if __name__ == "__main__":
    detect()
