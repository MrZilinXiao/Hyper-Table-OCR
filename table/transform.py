import numpy as np
import cv2
import imutils
from table.HED import predict_edge


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def preprocess(ori_img, hed_model=None):
    # 1. edge detection
    tmp = ori_img.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if hed_model is None:
        edged = cv2.Canny(gray, 75, 200)
    else:  # refer to HED model
        edged = predict_edge(hed_model, ori_img).astype(np.uint8)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # keep easy format
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
        cv2.drawContours(tmp, [screenCnt], -1, (0, 255, 0), 2)
        warped = four_point_transform(ori_img, screenCnt.reshape(4, 2))
        return warped, tmp
    else:
        raise RuntimeError("没有可见清晰边缘，透视变换失败")


if __name__ == '__main__':
    from table.HED import HEDNet

    model = HEDNet()
    model.load_weight()

    ori_img = cv2.imread('./test_720p.JPG')
    warped, flagged = preprocess(ori_img, hed_model=model)
    cv2.imwrite('warped.jpg', warped)
    cv2.imwrite('flagged.jpg', flagged)
