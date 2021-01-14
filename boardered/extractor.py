from abc import ABC
from typing import List

import cv2 as cv
import numpy as np

import utils
from table import TraditionalTable
from skimage import measure
from utils import Timer, draw_lines, minAreaRectbox, RemoteLogger
from boardered.table_net import table_line


class CellExtractor(ABC):
    """
    A unified interface for boardered extractor.
    OpenCV & UNet Extractor can derive from this interface.
    """

    def __init__(self):
        pass

    def get_cells(self, ori_img, table_coords) -> List[np.ndarray]:
        """
        :param ori_img: original image
        :param table_coords: List[np.ndarray], xyxy coord of each table
        :return: List[np.ndarray], [[xyxyxyxy(cell1), xyxyxyxy(cell2)](table1), ...]
        """
        pass


class TraditionalExtractor(CellExtractor):
    """
    This extractor requires extreme standard table image! Use with caution!
    'max_threshold_value', 'block_size', 'threshold', 'scale' can be altered throught __init__()
    """
    MAX_THRESHOLD_VALUE = 255
    BLOCK_SIZE = 15
    THRESHOLD_CONSTANT = 0
    SCALE = 15
    _AVAIL_PARAMS = {'max_threshold_value', 'block_size', 'threshold', 'scale'}

    def __init__(self, *kw, **kwargs):
        super().__init__()
        for param in kwargs:
            if param in self._AVAIL_PARAMS:
                self.__dict__[param.upper()] = kwargs[param]
            else:
                raise KeyError("没有%s这个参数可供调整！" % param)

    def get_cells(self, ori_img, table_coords, debug=False) -> List[np.ndarray]:
        # raise RuntimeError("此模块调试未完成！请选择其他获取表格框的方法！")
        cells = []
        for coord in table_coords:  # for each boarded table
            table_cell = []
            xmin, ymin, xmax, ymax = [int(k) for k in coord]  # used for cropping & shifting
            table_img = ori_img[ymin:ymax, xmin:xmax]
            with utils.Timer("Traditional Cell Detection"):
                grayscale = cv.cvtColor(table_img, cv.COLOR_BGR2GRAY)
                filtered = cv.adaptiveThreshold(~grayscale, self.MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C,
                                                cv.THRESH_BINARY,
                                                self.BLOCK_SIZE, self.THRESHOLD_CONSTANT)

                if debug:
                    cv.namedWindow('filtered', 0)
                    cv.resizeWindow('filtered', 900, 700)
                    cv.imshow('filtered', filtered)
                    cv.waitKey(0)

                horizontal = filtered.copy()
                vertical = filtered.copy()
                horizontal_size = int(horizontal.shape[1] / self.SCALE)
                horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
                utils.isolate_lines(horizontal, horizontal_structure)

                vertical_size = int(vertical.shape[0] / self.SCALE)
                vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
                utils.isolate_lines(vertical, vertical_structure)

                mask = horizontal + vertical

                if debug:
                    cv.namedWindow('mask', 0)
                    cv.resizeWindow('mask', 900, 700)
                    cv.imshow('mask', mask)
                    cv.waitKey(0)

                (contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                intersections = cv.bitwise_and(horizontal, vertical)

                if debug:
                    cv.namedWindow('intersections', 0)
                    cv.resizeWindow('intersections', 900, 700)
                    cv.imshow('intersections', intersections)
                    cv.waitKey(0)

                for i in range(len(contours)):
                    # Verify that region of interest is a table
                    (rect, table_joints) = utils.verify_table(contours[i], intersections)
                    if rect == None or table_joints is None:
                        continue

                    # Create a new instance of a table
                    table = TraditionalTable(rect[0], rect[1], rect[2], rect[3])

                    # Get an n-dimensional array of the coordinates of the table joints
                    joint_coords = []
                    for j in range(len(table_joints)):
                        joint_coords.append(table_joints[j][0][0])
                    joint_coords = np.asarray(joint_coords)

                    # Returns indices of coordinates in sorted order
                    # Sorts based on parameters (aka keys) starting from the last parameter, then second-to-last, etc
                    # joint_coords:
                    # [[913 179], [695 179], [548 179], [285 179], [182 179],
                    # [ 72 179], [913 119], [695 119], [548 119], [285 119],
                    # [182 119], [ 72 119], [913  31], [695  31], [548  31],
                    # [457  31], [376  31], [285  31], [182  31], [ 72  31],
                    # [913   0], [695   0], [548   0], ... ]
                    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
                    joint_coords = joint_coords[sorted_indices]

                    # Store joint coordinates in the table instance
                    table.set_joints(joint_coords)

                    table_entries = table.get_table_entries()
                    for k in range(len(table_entries)):
                        row = table_entries[k]
                        for j in range(len(row)):
                            entry = row[j]  # xyxy
                            table_cell.append(
                                [entry[0], entry[1], entry[2], entry[1], entry[2], entry[3], entry[0], entry[3]])  # xyxyxyxy
            cells.append(np.array(table_cell))

        return cells


class UNetExtractor(CellExtractor):
    def __init__(self, model, shape, hprob=0.5, vprob=0.5):
        super(UNetExtractor, self).__init__()
        self.model = model
        self.shape = shape
        self.hprob = hprob
        self.vprob = vprob

    def get_cells(self, ori_img, table_coords) -> List[np.ndarray]:  # in tensorflow
        cells = []
        for coord in table_coords:  # for each boarded table
            xmin, ymin, xmax, ymax = [int(k) for k in coord]  # used for cropping & shifting
            table_img = ori_img[ymin:ymax, xmin:xmax]  # cropped img
            with Timer("lines extraction(in cells extraction)"):
                row_boxes, col_boxes = table_line(self.model, table_img[..., ::-1],
                                                  size=self.shape,
                                                  hprob=self.hprob,
                                                  vprob=self.vprob)
            tmp = np.zeros(ori_img.shape[:2], dtype=np.uint8)
            tmp = draw_lines(tmp, row_boxes + col_boxes, color=255, lineW=2)
            labels = measure.label(tmp < 255, connectivity=2)  # 解八连通区域
            regions = measure.regionprops(labels)
            cell_boxes = minAreaRectbox(regions, flag=False, W=tmp.shape[1], H=tmp.shape[0], filtersmall=True,
                                        adjustBox=True)
            cell_boxes = np.array(cell_boxes)
            if len(cell_boxes.shape) == 1:
                # TODO: Add Prompt
                RemoteLogger.info("在此表中未构建出cell！")
                continue
            # shifting to fit original image
            cell_boxes[:, [0, 2, 4, 6]] += xmin  # cell_boxes: [N, 8]  N: number of boxes of each table
            cell_boxes[:, [1, 3, 5, 7]] += ymin

            # sort boxes to avoid displacement
            # cell_boxes = np.array(utils.sorted_boxes(cell_boxes.reshape(-1, 4, 2))).reshape(-1, 8)

            cells.append(cell_boxes)
        return cells


if __name__ == '__main__':
    # unittest for TraditionalExtractor
    extractor = TraditionalExtractor()
    o_img = cv.imread('merged.jpg')
    o_img, degree = utils.eval_angle(o_img)
    print(degree)
    table_coord = np.array([[0, 0, *o_img.shape[:2][::-1]]])
    cells = extractor.get_cells(ori_img=o_img, table_coords=table_coord)

    drawn = utils.draw_boxes(o_img, cells[0])
    cv.imshow('drawn', drawn)
    cv.waitKey(0)
    pass