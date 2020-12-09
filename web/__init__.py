import xlsxwriter

from utils import Singleton, Timer, EmptyTimer, draw_lines, minAreaRectbox
from typing import Dict, List
from skimage import measure
from TableUNet import table_net, table_line
from table import Table, TableCell
import numpy as np
import yaml


@Singleton
class WebHandler:
    _DETECTION_MODEL = None
    _LINE_MODEL = None
    _OCR_MODEL = None

    def __init__(self, config_path='./config.yml', debug=False):
        self.debug = debug
        self.timer = Timer if self.debug else EmptyTimer
        self.config_dict: dict = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.hprob, self.vprob = self.config_dict['table_line']['hprob'], self.config_dict['table_line']['vprob']
        self.unet_shape = self.config_dict['table_line']['input_shape']
        # {'table_line': {'model_path': './table-line.h5', 'input_shape': [512, 512]}, 'ocr': {'model_path': ''}, 'web': {'host': '0.0.0.0', 'port': 23333}}
        if self._DETECTION_MODEL is None:
            pass  # TODO: load MMdetection
        if self._LINE_MODEL is None:
            self._LINE_MODEL = table_net((*self.config_dict['table_line']['input_shape'], 3), num_classes=2)
            self._LINE_MODEL.load_weights(self.config_dict['table_line']['model_path'])
        if self._OCR_MODEL is None:
            pass  # TODO: OCR wait implementation

    def pipeline(self, ori_img: np.ndarray):
        # 0. TODO: table adjustment for perspective transformation (pending)

        # 1. table detection for boarded & boarderless table
        with self.timer("table detection"):
            tables = self._get_tables(ori_img)  # table_coords not changed
        # 2. get cells using UNet and postprocessing
        with self.timer("cells extraction"):
            cells = self._get_cells(ori_img, tables)  # cells coords not changed
        # 3. get content from OCR (can be async if performance not fit)
        with self.timer("OCR pipeline"):
            ocr = self._get_ocr(ori_img, tables)  # ocr coords not changed
        # 4. match OCR content with cells
        with self.timer("match"):
            self._match(cells, ocr, tables)

    def _get_tables(self, ori_img) -> np.ndarray:  # xyxy
        return np.array([[0, 0, *ori_img.shape[1::-1]]])  # TODO: MMDetection Integration

    def _get_ocr(self, ori_img, table_coords) -> dict:
        pass  # TODO: OCR Integration

    def _get_cells(self, ori_img, table_coords) -> List[np.ndarray]:
        cells = []
        # table_imgs = []
        for coord in table_coords:  # for each boarded table
            xmin, ymin, xmax, ymax = coord  # used for cropping & shifting
            table_img = ori_img[ymin:ymax, xmin:xmax]  # cropped img
            row_boxes, col_boxes = table_line(table_img[..., ::-1],  # BGR->RGB
                                              size=self.unet_shape,
                                              hprob=self.hprob,
                                              vprob=self.vprob)
            tmp = np.zeros(ori_img.shape[:2], dtype=np.uint8)
            tmp = draw_lines(tmp, row_boxes + col_boxes, color=255, lineW=2)
            labels = measure.label(tmp < 255, connectivity=2)  # 解八连通区域
            regions = measure.regionprops(labels)
            cell_boxes = minAreaRectbox(regions, flag=False, W=tmp.shape[1], H=tmp.shape[0], filtersmall=True,
                                        adjustBox=True)
            cell_boxes = np.array(cell_boxes)
            # shifting to fit original image
            cell_boxes[:, [0, 2, 4, 6]] += xmin  # cell_boxes: [N, 8]  N: number of boxes of each table
            cell_boxes[:, [1, 3, 5, 7]] += ymin
            cells.append(cell_boxes)
        return cells

    def _match(self, cells, ocr: dict, tables):
        """
        对此我们的思路是列举所有的单元格候选，每个单元格表示为（起始行，结束行，起始列，结束列），
        然后对所有单元格按面积从小到大排序。接着遍历排序好的候选单元格，去判断其上下左右的框线
        是否都真实存在，若存在，则此单元格就在原图存在。注意到，每当确立一个单元格存在，所有与
        其共享起始行和起始列的其他单元格则不可能再存在，因为我们不考虑单元格中套着单元格的情况。
        所以虽然单元格候选集很大，但我们可以利用这一性质在遍历过程中进行剪枝，所以会很高效。
        翻译：
        1.
        """
        # 这里假设表格是正对的 但最好还是用xyxyxyxy
        assert len(cells) == len(tables), "table & cell not match!"
        self.tables = []
        for i in range(len(tables)):
            my_table = Table(coord=tables[i], cells=cells[i], verbose=self.debug)
            my_table.match_ocr(ocr)
            my_table.build_structure()
            self.tables.append(my_table)

    def to_excel(self, filename='./debug.xlsx'):
        workbook = xlsxwriter.Workbook(filename)
        for (i, table) in enumerate(self.tables):
            worksheet = workbook.add_worksheet(name='识别表格' + str(i))
            for (row_num, row) in enumerate(table.rows):
                row: List[TableCell]
                for cell in row:
                    worksheet.merge_range(row_num, cell.col_range[0], row_num, cell.col_range[1], data=cell.ocr_content)
        workbook.close()