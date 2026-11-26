import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('IMG_20261111_173600995_HDR.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# low_threshold = 50
# high_threshold = 200
# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 50  # minimum number of pixels making up a line
# max_line_gap = 20  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img) * 0  # creating a blank to draw lines on

# # Run Hough on edge detected image
# # Output "lines" is an array containing endpoints of detected line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                     min_line_length, max_line_gap)

# for line in lines:
#     x1,y1,x2,y2 = line
#     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# # Draw the lines on the  image
# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

# plt.imshow(lines_edges)
# plt.show()

# 1. Load image in grayscale and reduce noise
# img = cv2.imread('ellipses.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 2. Detect edges (or use cv2.threshold for high-contrast binary shapes)
# edges = cv2.Canny(blurred, 50, 150)

# # 3. Find contours
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     # fitEllipse requires at least 5 points
#     if len(cnt) >= 5:
#         # Optional: Filter out tiny noise contours by area
#         if cv2.contourArea(cnt) < 100:
#             continue
            
#         # 4. Fit the ellipse
#         # Returns ((center_x, center_y), (major_axis, minor_axis), angle)
#         ellipse = cv2.fitEllipse(cnt)
        
#         # Draw the detected ellipse in green
#         cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# cv2.imshow('Detected Ellipses', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect ellipses directly using edge-drawing / arc-grouping
# Returns a list of vectors: [x_center, y_center, a_axis, b_axis, angle, score]
ellipses = cv2.ximgproc.findEllipses(
    gray,
    scoreThreshold=0.7,      # Minimum confidence score (0.0 to 1.0)
    reliabilityThreshold=0.5, # Minimum arc reliability
    centerDistanceThreshold=0.05 # Distance threshold for grouping arcs
)

overlay = np.zeros_like(img)
if ellipses is not None:
    for ellipse in ellipses:
        cx, cy, a, b, angle, score = ellipse
        
        # Format into ((center), (axes), angle) for cv2.ellipse
        # center = (int(cx), int(cy))
        # axes = (int(a), int(b))
        
        cv2.ellipse(overlay, ((cx, cy), (a, b), angle), (255, 0, 0), 2)

# cv2.imshow('ximgproc Ellipses', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_overlay = cv2.addWeighted(img, 0.8, overlay, 1, 0)
plt.imshow(img_overlay)
plt.show()