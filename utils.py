import datetime
import logging
import os
import random
import sys
import time

import cv2 as cv
import numpy as np
import scipy.spatial.distance as dist
from PIL import Image
from skimage import measure

MIN_TABLE_AREA = 50  # min table area to be considered a table
EPSILON = 3  # epsilon value for contour approximation


class LineCluster(object):
    def __init__(self, max_size=100):
        self.curr = 0
        self.lines = np.zeros((max_size, 4))

    def add_line(self, line):
        self.lines[self.curr] = line
        self.curr += 1

    def reset(self, max_size=100):
        self.__init__(max_size)

    def get_cluster_line(self) -> (float, float, float, float):
        return (np.average(self.lines[0:self.curr, i]) for i in range(4))


class CVHelper:
    @classmethod
    def _is_close_line_pair(cls, line1, line2, alpha=0.1, degree=np.pi / 18):
        """
        :param line1: xyxy
        :param line2: xyxy
        :param alpha: float, used for judging whether two parallel
        :return: bool, whether two lines are closed
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        len1 = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        len2 = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3))
        product = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)  # vector product
        if np.fabs(product / len1 * len2) < np.cos(degree):
            return False
        mx1, mx2 = (x1 + x2) / 2, (x3 + x4) / 2
        my1, my2 = (y1 + y2) / 2, (y3 + y4) / 2
        dist = np.sqrt((mx1 - mx2) * (mx1 - mx2) + (my1 - my2) * (my1 - my2))
        if dist > max(len1, len2) * alpha:
            return False
        return True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def isolate_lines(src, structuring_element):
    cv.erode(src, structuring_element, src, (-1, -1))  # makes white spots smaller
    cv.dilate(src, structuring_element, src, (-1, -1))  # makes white spots bigger


def verify_table(contour, intersections):
    area = cv.contourArea(contour)

    if area < MIN_TABLE_AREA:
        return (None, None)

    # approxPolyDP approximates a polygonal curve within the specified precision
    curve = cv.approxPolyDP(contour, EPSILON, True)

    # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
    rect = cv.boundingRect(curve)  # format of each rect: x, y, w, h

    # Finds the number of joints in each region of interest (ROI)
    # Format is in row-column order (as finding the ROI involves numpy arrays)
    # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    (possible_table_joints, _) = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Determines the number of table joints in the image
    # If less than 5 table joints, then the image
    # is likely not a table
    if len(possible_table_joints) < 5:
        return (None, None)

    return rect, possible_table_joints


class EmptyTimer(object):
    def __init__(self, *kw, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Timer(object):
    def __init__(self, reason, verbose=True):
        self.verbose = verbose
        if self.verbose:
            self.reason = reason

    def __enter__(self):
        self.st_time = time.time()
        return self

    def __exit__(self, *args):
        self.secs = time.time() - self.st_time
        self.msecs = self.secs * 1000
        if self.verbose:
            print('%s consumes %.2f ms' % (self.reason, self.msecs))


def _order_points(pts):
    # 根据x坐标对点进行排序
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def image_location_sort_box(box):
    """
    为xyxyxyxy的bbox的点的顺序排序 top-left, top-right, bottom-right, bottom-left
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def minAreaRect(coords):
    """
    多边形外接矩形
    """
    rect = cv.minAreaRect(coords[:, ::-1])  # 包围二维点集的最小倾斜矩形
    box = cv.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = image_location_sort_box(box)

    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2

    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2
    # degree,w,h,cx,cy = solve(box)
    # x1,y1,x2,y2,x3,y3,x4,y4 = box
    # return {'degree':degree,'w':w,'h':h,'cx':cx,'cy':cy}
    return [xmin, ymin, xmax, ymax]


def get_table_line(binimg, axis=0, lineW=10):
    ##获取表格线
    ##axis=0 横线
    ##axis=1 竖线
    labels = measure.label(binimg > 0, connectivity=2)  # 8连通区域标记
    regions = measure.regionprops(labels)  #
    if axis == 1:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > lineW]
    else:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > lineW]
    return lineboxes


def draw_lines(im, bboxes, color=(255, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]

    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        cv.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv.LINE_AA)

    return tmp


def letterbox_image(image, size, fillValue=[128, 128, 128]):
    """
    resize image with unchanged aspect ratio using padding
    """
    image_h, image_w = image.shape[:2]
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    # cv2.imwrite('tmp/test.png', resized_image[...,::-1])
    if fillValue is None:
        fillValue = [int(x.mean()) for x in cv.split(np.array(image))]
    boxed_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    boxed_image[:] = fillValue
    boxed_image[:new_h, :new_w, :] = resized_image

    return boxed_image, new_w / image_w, new_h / image_h


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def adjust_lines(RowsLines, ColsLines, alph=50):
    ##调整line

    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newRowsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newRowsLines.append([x2, y2, x4, y4])

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    if r < alph:
                        newColsLines.append([x1, y1, x3, y3])
                    r = sqrt((x1, y1), (x4, y4))
                    if r < alph:
                        newColsLines.append([x1, y1, x4, y4])

                    r = sqrt((x2, y2), (x3, y3))
                    if r < alph:
                        newColsLines.append([x2, y2, x3, y3])
                    r = sqrt((x2, y2), (x4, y4))
                    if r < alph:
                        newColsLines.append([x2, y2, x4, y4])

    return newRowsLines, newColsLines


def fit_line(p1, p2):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def point_line_cor(p, A, B, C):
    # 判断点与之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r


def line_to_line(points1, points2, alpha=10):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    A1, B1, C1 = fit_line((x1, y1), (x2, y2))  # AX+BY+C=0
    A2, B2, C2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], A2, B2, C2)  # r = flag
    flag2 = point_line_cor([x2, y2], A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):  # L1两点分别在线的两边

        x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
        y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
        p = (x, y)
        r0 = sqrt(p, (x1, y1))
        r1 = sqrt(p, (x2, y2))

        if min(r0, r1) < alpha:

            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]

    return points1


from numpy import cos, sin


def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


def xy_rotate_box(cx, cy, w, h, angle=0, degree=None, **args):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
    if degree is not None:
        angle = degree
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


def minAreaRectbox(regions, flag=True, W=0, H=0, filtersmall=False, adjustBox=False):
    """
    多边形外接矩形
    """
    boxes = []
    for region in regions:
        rect = cv.minAreaRect(region.coords[:, ::-1])

        box = cv.boxPoints(rect)
        box = box.reshape((8,)).tolist()
        box = image_location_sort_box(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        angle, w, h, cx, cy = solve(box)
        if adjustBox:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w + 5, h + 5, angle=0, degree=None)

        if w > 32 and h > 32 and flag:
            if abs(angle / np.pi * 180) < 20:
                if filtersmall and w < 10 or h < 10:
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            if w * h < 0.5 * W * H:
                if filtersmall and w < 8 or h < 8:
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return boxes


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv.resize(im, (0, 0), fx=f, fy=f)


from numpy import amin, amax
from scipy.ndimage import filters, interpolation


def estimate_skew_angle(raw, angleRange=[-15, 15]):
    """
    估计图像文字偏转角度,
    angleRange:角度估计区间
    """
    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw - amin(raw)
    image = image / amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)
    # w,h = image.shape[1],image.shape[0]
    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w] - m[:h, :w] + 1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = amax(flat) - flat
    flat -= amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(angleRange[0], angleRange[1])
    estimates = []
    for a in angles:
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))

    _, a = max(estimates)
    return a


def eval_angle(img, angleRange=[-5, 5]):
    """
    估计图片文字的偏移角度
    """
    im = Image.fromarray(img)
    degree = estimate_skew_angle(np.array(im.convert('L')), angleRange=angleRange)
    im = im.rotate(degree, center=(im.size[0] / 2, im.size[1] / 2), expand=1, fillcolor=(255, 255, 255))
    img = np.array(im)
    return img, degree


def draw_boxes(im, bboxes, color=(0, 0, 0)):
    """
        boxes: bounding boxes
    """
    color_candidates = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    tmp = np.copy(im)
    h, w, _ = im.shape

    for box in bboxes:
        if type(box) is dict:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(**box)
        else:
            if len(box) == 4:
                _x1, _y1, _x2, _y2 = box[:4]
                h = _y2 - _y1
                w = _x2 - _x1
                x1, y1, x2, y2, x3, y3, x4, y4 = _x1, _y1, _x1 + w, _y1, _x1 + w, _y1 + h, _x1, _y1 + h
            else:
                x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        c = random.choice(color_candidates)
        cv.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1, lineType=cv.LINE_AA)
        cv.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1, lineType=cv.LINE_AA)
        cv.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1, lineType=cv.LINE_AA)
        cv.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1, lineType=cv.LINE_AA)

    return tmp


def Singleton(cls):
    """
    A decorator for Singleton support
    """
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class MyLogger(logging.Logger):
    def __init__(self, level='DEBUG', file='web_remote.log'):
        super().__init__(file)
        self.setLevel(level)
        ft = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setFormatter(ft)
            self.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = ft
        self.addHandler(console_handler)
        self.propagate = False


# Using a singleton decorator seems less elegant
# @Singleton
# class RemoteLogger(object):
#     def __init__(self, line_limit=100, debug=False):
#         self.line_limit = line_limit
#         self.debug = debug
#         self.lines = []
#
#     def info(self, line):
#         self.lines.append('<li>' + line + '</li>')
#         if self.debug:
#             print(line)
#         self.lines = self.lines[-self.line_limit:]

class RemoteLogger(object):
    lines = []
    debug = True
    line_limit = 100

    @classmethod
    def _prefix(cls, _format='%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.strftime(datetime.datetime.now(), _format)

    @classmethod
    def info(cls, line):
        cls.lines.append('<li>' + cls._prefix() + ' ' + line + '</li>')
        if cls.debug:
            print(line)
        cls.lines = cls.lines[-cls.line_limit:]


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], \
                          [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv.getPerspectiveTransform(points, pts_std)
    dst_img = cv.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
