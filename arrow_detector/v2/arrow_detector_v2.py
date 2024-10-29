import cv2
import numpy as np
import math

def preprocess(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 3)
    canny_img = cv2.Canny(blur_img, 50, 100)
    filter = np.ones((3, 3))
    dilate_img = cv2.dilate(canny_img, filter, iterations=3)
    erode_img = cv2.erode(dilate_img, filter, iterations=2)
    return erode_img


def rad_to_deg(rad):
    deg = rad*360/(2*math.pi)
    return deg

def angle(tip, mid):
    tip_mod = tip.copy()
    mid_mod = mid.copy()
    tip_mod[1] = y - tip_mod[1]
    mid_mod[1] = y - mid_mod[1]
    slope = (tip_mod[1] - mid_mod[1]) / (tip_mod[0] - mid_mod[0])
    degrees = math.atan(slope)

    if (tip_mod[0] < mid_mod[0]):
        degrees = degrees*360/(2*math.pi) + 90
    elif (tip_mod[0] > mid_mod[0]):
        degrees = degrees*360/(2*math.pi) - 90
    elif (tip_mod[0] == mid_mod[0]):
        if (tip_mod[1] < mid_mod[1]):
            degrees = 180
        if (tip_mod[1] > mid_mod[1]):
            degrees = 0

    print("The arrow is at an angle of ", degrees, "degrees")
    return degrees

img = cv2.imread("arrow_r.jpg")
x = img.shape[1]
y = img.shape[0]

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([5, 50, 50])
upper_yellow = np.array([30, 255, 255])
mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
mask_img = cv2.bitwise_and(img, img, mask=mask_yellow)

contours, hierarchy = cv2.findContours(preprocess(mask_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

peri = cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], 0.05 * peri, True)

cv2.drawContours(img, contours[0], -1, (0, 255, 255), 3)

m = cv2.moments(contours[0])
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])
com = [cx, cy]
cv2.circle(img, com, 1, (255, 0, 0), 3)

point_index = {
    0 : 0,
    1 : 0,
    2 : 0
}

length_list = []

for i in range(3):
    length = math.sqrt(math.pow(float((approx[(i+2)%3][0][1]))-float(approx[(i+1)%3][0][1]), 2) + math.pow(float((approx[(i+2)%3][0][0]))-float(approx[(i+1)%3][0][0]), 2))
    length_list.append(length)
    point_index.update({i : length})

length_list.sort()
largest_length = length_list[-1]
value_list = list(point_index.values())
tip_index = value_list.index(largest_length)

tip = [approx[tip_index][0][0], approx[tip_index][0][1]]
angle = angle(tip, com)

cv2.circle(img, tip, 1, (255, 0, 0), 3)
cv2.line(img, com, (com[0], 0), (200, 0, 200), 2)
cv2.line(img, com, tip, (200, 0, 200), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
