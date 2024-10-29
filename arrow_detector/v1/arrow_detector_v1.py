import cv2
import numpy
import math

def preprocess(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray_img)
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 6)
    # cv2.imshow("blur", blur_img)
    # th1, thresh_img = cv2.threshold(blur_img, 125, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", thresh_img)
    canny_img = cv2.Canny(blur_img, 50, 100)
    # cv2.imshow("canny", canny_img)
    filter = numpy.ones((3, 3))
    dilate_img = cv2.dilate(canny_img, filter, iterations=3)
    # cv2.imshow("dilated", dilate_img)
    erode_img = cv2.erode(dilate_img, filter, iterations=2)
    # cv2.imshow("eroded", erode_img)
    return erode_img

def find_tip(points, convex_hull):
    length = len(points)
    indices = numpy.setdiff1d(range(length), convex_hull)

    if len(indices) == 2:
        if numpy.all(points[(indices[0]+2)%length] == points[(indices[1]-2)%length]):
            tip_index = (indices[0]+2)%length
            base_1 = list(points[(tip_index+3)%length])
            base_2 = list(points[(tip_index-3)%length])
            mid = [int((base_1[0]+base_2[0])/2), int((base_1[1]+base_2[1])/2)]
            return list(points[tip_index]), mid
        if numpy.all(points[(indices[1] + 2) % length] == points[(indices[0] - 2) % length]):
            tip_index = (indices[1] + 2)%length
            base_1 = list(points[(tip_index+3)%length])
            base_2 = list(points[(tip_index-3)%length])
            mid = [int((base_1[0]+base_2[0])/2), int((base_1[1]+base_2[1])/2)]
            return list(points[tip_index]), mid

    elif len(indices) == 1:
        if((indices[0] + 3) % 6 == (indices[0] - 3) % 6):
            tip_index = (indices[0] + 3) % length
            mid = list(points[(tip_index + 3) % length])
            return list(points[tip_index]), mid

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

img = cv2.imread("arrows2_l.jpg")
x = img.shape[1]
y = img.shape[0]

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red_1 = numpy.array([0, 50, 50])
upper_red_1 = numpy.array([10, 255, 255])
lower_red_2 = numpy.array([170, 50, 50])
upper_red_2 = numpy.array([179, 255, 255])
mask_red_1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
mask_red_2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)
mask_img_1 = cv2.bitwise_and(img, img, mask=mask_red_1)
mask_img_2 = cv2.bitwise_and(img, img, mask=mask_red_2)
final_mask_img = cv2.bitwise_or(mask_img_1, mask_img_2)

contours, hierarchy = cv2.findContours(preprocess(final_mask_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    hull = cv2.convexHull(approx, returnPoints=False)  # Returns only the indexes of the approx array which are in the convex hull
    sides = len(hull)
    area = cv2.contourArea(cnt)

    if (6 > sides > 3) and (((sides + 2) == len(approx)) or ((sides+1)==len(approx))): #and area > 750:
        arrow_tip, mid = find_tip(approx[:,0,:], hull.squeeze())
        angle(arrow_tip, mid)
        if arrow_tip:
            cv2.drawContours(img, [cnt], -1, (200, 255, 0), 3)
            cv2.circle(img, arrow_tip, 3, (0, 130, 255), cv2.FILLED)
            cv2.line(img, arrow_tip, mid, (200, 150, 255), 2)
            cv2.line(img, [mid[0], 0], mid, (200, 150, 255), 2)

img_resize = cv2.resize(img, (0, 0),  fx=0.75, fy=0.75)
cv2.imshow("Image", img_resize)
cv2.waitKey(0)

