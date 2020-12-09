from typing import List, Union, Tuple, Iterator
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import itertools
from bisect import bisect_left
import cv2
from utils import draw_boxes


# class Table:
#     def __init__(self, x, y, w, h):
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.joints = None
#
#     def __repr__(self):
#         return "(x: %d, y: %d, w: %d, h: %d)" % (self.x, self.x + self.w, self.y, self.y + self.h)
#
#     # Stores the coordinates of the table joints.
#     # Assumes the n-dimensional array joints is sorted in ascending order.
#     def set_joints(self, joints):
#         if self.joints != None:
#             raise ValueError("Invalid setting of table joints array.")
#
#         self.joints = []
#         row_y = joints[0][1]
#         row = []
#         for i in range(len(joints)):
#             if i == len(joints) - 1:
#                 row.append(joints[i])
#                 self.joints.append(row)
#                 break
#
#             row.append(joints[i])
#
#             # If the next joint has a new y-coordinate,
#             # start a new row.
#             if joints[i + 1][1] != row_y:
#                 self.joints.append(row)
#                 row_y = joints[i + 1][1]
#                 row = []
#
#     # Prints the coordinates of the joints.
#     def print_joints(self):
#         if self.joints == None:
#             print("Joint coordinates not found.")
#             return
#
#         print("[")
#         for row in self.joints:
#             print("\t" + str(row))
#         print("]")
#
#     # Finds the bounds of table entries in the image by
#     # using the coordinates of the table joints.
#     def get_table_entries(self):
#         if self.joints == None:
#             print("Joint coordinates not found.")
#             return
#
#         entry_coords = []
#         for i in range(0, len(self.joints) - 1):
#             entry_coords.append(self.get_entry_bounds_in_row(self.joints[i], self.joints[i + 1]))
#
#         return entry_coords
#
#     # Finds the bounds of table entries
#     # in each row based on the given sets of joints.
#     def get_entry_bounds_in_row(self, joints_A, joints_B):
#         row_entries = []
#
#         # Since the sets of joints may not have the same
#         # number of points, we pick the set with a lower number
#         # of points to find the bounds from.
#         if len(joints_A) <= len(joints_B):
#             defining_bounds = joints_A
#             helper_bounds = joints_B
#         else:
#             defining_bounds = joints_B
#             helper_bounds = joints_A
#
#         for i in range(0, len(defining_bounds) - 1):
#             x = defining_bounds[i][0]
#             y = defining_bounds[i][1]
#             w = defining_bounds[i + 1][0] - x  # helper_bounds's (i + 1)th coordinate may not be the lower-right corner
#             h = helper_bounds[0][1] - y  # helper_bounds has the same y-coordinate for all of its elements
#
#             # If the calculated height is less than 0,
#             # make the height positive and
#             # use the y-coordinate of the row above for the bounds
#             if h < 0:
#                 h = -h
#                 y = y - h
#
#             row_entries.append([x, y, w, h])
#
#         return row_entries

class OCRBlock(object):
    def __init__(self, coord, content, conf=-1.0):
        self.coord: np.ndarray = coord  # xyxyxyxy
        self.conf = conf
        assert len(coord) == 8, "xyxyxyxy not fit for OCRBlock!"
        self.shape = Polygon([coord[0:2], coord[2:4], coord[4:6], coord[6:]])
        self.ocr_content: Union[List[str], str] = content


class TableCell(OCRBlock):
    def __init__(self, coord):
        super(TableCell, self).__init__(coord, [])
        self.matched = False
        self.row_range = [-1, -1]
        self.col_range = [-1, -1]

    @property
    def upper_y(self):
        return self.coord[[1, 3]].mean()

    @property
    def left_x(self):
        return self.coord[[0, 6]].mean()

    @property
    def right_x(self):
        return self.coord[[2, 4]].mean()


class Table(object):
    def __init__(self, coord, cells, verbose=True):
        """
        :param verbose: whether to show progress
        """
        self.verbose = verbose
        self.row_num, col_num = -1, -1
        self.rows: List[List[TableCell]] = []
        self.coord = coord  # xyxy
        self.cells: np.ndarray = cells  # ordered
        self._load_from_cells()

    def _load_from_cells(self):
        self.table_cells = [TableCell(row) for row in self.cells]

    def _load_ocr_dict(self, ocr) -> List[OCRBlock]:
        ocr_block_list = []
        for entry in ocr['sentences']:
            entry: List[str, List[List[int]], int]
            # ["Test", [[], [], [], []], 0.95]
            _str, coord, conf = entry
            coord = list(itertools.chain(*coord))
            ocr_block_list.append(OCRBlock(coord, _str, conf=conf))
        # TODO: sort here to maximum performance
        return ocr_block_list

    @classmethod
    def _closest_idx(cls, _list, num):
        left_idx = bisect_left(_list, num)
        if left_idx == 0:
            return 0
        if left_idx == len(_list):
            return left_idx - 1
        return left_idx - 1 if num - _list[left_idx - 1] <= _list[left_idx] - num else left_idx

    def match_ocr(self, ocr: dict, thresh=0.95):
        if self.verbose:
            xmin, ymin, xmax, ymax = self.coord
            empty = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
            cv2.namedWindow('Show OCR Match')
        # from dict to OCRBlock objects
        self.ocr_blocks: List[OCRBlock] = self._load_ocr_dict(ocr)
        for table_cell in self.table_cells[::-1]:  # traverse in reserved order
            for ocr_block in self.ocr_blocks[::-1]:
                if self.verbose:
                    tmp = draw_boxes(empty, [table_cell.coord])
                    tmp = draw_boxes(tmp, [ocr_block.coord])
                    cv2.imshow('Show OCR Match', tmp)
                    cv2.waitKey(0)

                if table_cell.shape.intersection(ocr_block.shape).area / ocr_block.shape.area >= thresh:
                    # matched!
                    table_cell.ocr_content.append(ocr_block.ocr_content)
                    table_cell.matched = True
                    self.ocr_blocks.remove(ocr_block)

    def build_structure(self, delta_y=10, delta_x=10):
        """
        对此我们的思路是列举所有的单元格候选，每个单元格表示为（起始行，结束行，起始列，结束列），
        然后对所有单元格按面积从小到大排序。接着遍历排序好的候选单元格，去判断其上下左右的框线
        是否都真实存在，若存在，则此单元格就在原图存在。注意到，每当确立一个单元格存在，所有与
        其共享起始行和起始列的其他单元格则不可能再存在，因为我们不考虑单元格中套着单元格的情况。
        所以虽然单元格候选集很大，但我们可以利用这一性质在遍历过程中进行剪枝，所以会很高效。
        步骤：
        0. 给出可能的最大行、最大列
        1. self.table_cells 按面积从小到大排序
        2.
        """
        # 0. assign cells in rows by different y (average y of two points from upper edge)
        if len(self.table_cells) == 0:
            raise ValueError("No available cells! ")
        row_y = self.table_cells[0].upper_y
        row = []
        for i in range(len(self.table_cells)):
            if i == len(self.table_cells) - 1:  # last cell, should break
                row.append(self.table_cells[i])
                self.rows.append(row)
                break
            row.append(self.table_cells[i])
            if not np.isclose(row_y, self.table_cells[i + 1].upper_y, atol=delta_y):  # go for a new row
                self.rows.append(row)
                row = []
            row_y = self.table_cells[i + 1].upper_y  # update latest row_y

        # 1. build table through greedy searching
        self.rows: np.ndarray[List[TableCell]] = np.array(self.rows)  # go numpy for list-index
        rows_num = np.array([len(row) for row in self.rows])
        self.row_num = np.max(rows_num)
        merged_rows_idx = np.where(rows_num < self.row_num)
        complete_rows_idx = np.where(rows_num == self.row_num)[0].tolist()

        # set row_range (temporary)
        for row_num, row in enumerate(self.rows):
            for cell in row:
                cell.row_range = [row_num, row_num]

        # calculate col_range by delta_x
        for k, merged_row in enumerate(self.rows[merged_rows_idx]):
            merged_row: List[TableCell]
            closest_complete_row_idx = self._closest_idx(complete_rows_idx, merged_rows_idx[k])  # find the closest row idx
            right_match = lambda k1, k2: np.isclose(merged_row[k1].right_x, self.rows[complete_rows_idx[closest_complete_row_idx]][k2].right_x,
                                                    atol=delta_x)
            left_idx = 0
            checkpoints = []
            for i in range(len(merged_row)):
                while not right_match(i, left_idx):
                    left_idx += 1
                checkpoints.append(left_idx)
                left_idx += 1
                if left_idx >= len(self.rows[complete_rows_idx[closest_complete_row_idx]]):
                    break

            assert len(merged_row) == len(checkpoints)

            if not checkpoints:
                break
            range_l, range_r = 0, -1
            for j in range(len(merged_row)):
                range_r = checkpoints[j]
                merged_row[j].col_range = [range_l, range_r]
                range_l = range_r + 1


if __name__ == '__main__':
    # Simple test for build_structure

    pass
    # a = [1, 2, 3, 4, 5]
    # b = [1, 2, 4, 6, 5]
    # for i in a[::-1]:
    #     for j in b[::-1]:
    #         if i == j:
    #             print("matched for %d" % i)
    #             b.remove(i)
    #             a.remove(i)
    #         print("now: a=%s, b=%s" % (a, b))
