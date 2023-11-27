import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

WARP_SRC = np.float32([(0.01, 0.83), (0.98, 0.78), (0.24, 0.4), (0.75, 0.4)])
WARP_DST = np.float32(
    [(0, 1 - 0.05), (1 - 0.05, 1 - 0.05), (0 + 0.05, 0 + 0.05), (1 - 0.05, 0)]
)
INV_WARP_SRC = WARP_DST
INV_WARP_DST = WARP_SRC


def detect(frame):
    img_clone = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    _, bin_img = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY)

    warp = perspective_warp(image=bin_img, image_to_draw_dots=frame.copy())

    canny = cv.Canny(warp, 120, 210)

    out_img, curves, lanes, plot_y = sliding_window_search(canny)

    curve_rad = get_curve(frame.copy(), curves[0], curves[1])
    print(curve_rad)
    img_with_lanes = draw_lanes(img_clone, curves[0], curves[1])

    cv.imshow("Warp", warp)
    cv.imshow("Original", frame)
    cv.imshow("Canny", canny)
    cv.imshow("Sliding Window Search", out_img)
    cv.imshow("Lanes", img_with_lanes)


def roi(image, region):
    pass


def perspective_warp(image, src=WARP_SRC, dst=WARP_DST, image_to_draw_dots=None):
    height, width, *_ = image.shape

    img_size = np.float32([width, height])

    src = src * img_size
    dst = dst * img_size

    if image_to_draw_dots is not None:
        for i, pt in enumerate(src):
            pt_int = np.int16(pt)
            cv.circle(image_to_draw_dots, pt_int, 5, (255, 0, 0), -1)
            cv.putText(
                image_to_draw_dots,
                f"{i}",
                pt_int,
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
            )

    transform_mat = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(
        image, transform_mat, (width, height), flags=cv.INTER_LANCZOS4
    )
    return warped


def inv_perspective_warp(
    image,
    src=INV_WARP_SRC,
    dst=INV_WARP_DST,
):
    height, width, *_ = image.shape
    img_size = np.float32([width, height])

    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * img_size
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(image, M, (width, height), flags=cv.INTER_LANCZOS4)
    return warped


def sliding_window_search(
    img, windows_number=9, margin=40, min_pix=1, draw_windows=True
):
    global left_a, left_b, left_c, right_a, right_b, right_c  # For quadratic equation
    left_a, left_b, left_c = [], [], []
    right_a, right_b, right_c = [], [], []

    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = np.sum(img[img.shape[0] // 2 :, :], axis=0)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int32(img.shape[0] / windows_number)
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
            left_x_current = np.int32(np.mean(nonzero_x[good_left_idx]))
        if len(good_right_idx) > min_pix:
            right_x_current = np.int32(np.mean(nonzero_x[good_right_idx]))

    # Concatenate the arrays of indices
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_idx]
    left_y = nonzero_y[left_lane_idx]
    right_x = nonzero_x[right_lane_idx]
    right_y = nonzero_y[right_lane_idx]

    print(
        f"x_left:{len(left_x)}; left_y:{len(left_y)}; right_x:{len(right_x)}; right_y:{len(right_y)}"
    )
    # Fit a second order polynomial to each
    if len(left_x) > 0 and len(right_x) > 0 and len(left_y) > 0 and len(right_y) > 0:
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
    else:
        right_fit = np.array([0, 0, 0])
        left_fit = np.array([0, 0, 0])

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    # Calculate the mean of 10 recent values for a,b,c
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit_x = left_fit_[0] * plot_y**2 + left_fit_[1] * plot_y + left_fit_[2]
    right_fit_x = right_fit_[0] * plot_y**2 + right_fit_[1] * plot_y + right_fit_[2]

    out_img[nonzero_y[left_lane_idx], nonzero_x[left_lane_idx]] = [255, 0, 100]
    out_img[nonzero_y[right_lane_idx], nonzero_x[right_lane_idx]] = [0, 100, 255]

    return out_img, (left_fit_x, right_fit_x), (left_fit_, right_fit_), plot_y


def get_curve(img, left_x, right_x):
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(plot_y)
    ym_per_pix = 30.5 / 720  # meters per pixel in y dimension (Tweak this)
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension (Tweak this)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curve_rad = (
        (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit_cr[0])
    right_curve_rad = (
        (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = (
        left_fit_cr[0] * img.shape[0] ** 2
        + left_fit_cr[1] * img.shape[0]
        + left_fit_cr[2]
    )
    r_fit_x_int = (
        right_fit_cr[0] * img.shape[0] ** 2
        + right_fit_cr[1] * img.shape[0]
        + right_fit_cr[2]
    )
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curve_rad, right_curve_rad, center)


def draw_lanes(img, left_fit, right_fit):
    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_image = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, plot_y]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, plot_y])))])
    points = np.hstack((left, right))

    cv.fillPoly(color_image, np.int_(points), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_image)
    inv_perspective = cv.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective


if __name__ == "__main__":
    detect()
