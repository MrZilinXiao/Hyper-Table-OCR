import tensorflow as tf

# from boardered.table_net import table_net
for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)
import xlsxwriter

from utils import Singleton, Timer, EmptyTimer, draw_lines, minAreaRectbox, draw_boxes
from typing import List
from skimage import measure
# from TableUNet import table_net
from boardered.table_net import table_line, table_net
from table import Table, TableCell
from table.transform import preprocess
from ocr import PaddleHandler
import uuid
import os

from mmdet.apis import init_detector, inference_detector
import numpy as np
import yaml
import torch
import cv2


@Singleton
class WebHandler:
    _DETECTION_MODEL = None
    _LINE_MODEL = None
    _OCR_MODEL = None
    _OCR_FACTORY = {
        'paddle': PaddleHandler
    }

    def __init__(self, config_path='./config.yml', debug=True, preload=True, device='cuda:0', static_folder='static/'):
        self.device = device
        if not tf.test.is_gpu_available():
            self.device = 'cpu'
        else:
            # self._configure_gpu()
            pass
        self.static_prefix = static_folder
        if not os.path.exists(static_folder):
            os.mkdir(static_folder)
        self.debug = debug
        self.timer = Timer if self.debug else EmptyTimer
        self.config_dict: dict = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        self.hprob, self.vprob = self.config_dict['table_line']['hprob'], self.config_dict['table_line']['vprob']
        self.unet_shape = self.config_dict['table_line']['input_shape']
        # {'table_line': {'model_path': './table-line.h5', 'input_shape': [512, 512]}, 'ocr': {'model_path': ''}, 'web': {'host': '0.0.0.0', 'port': 23333}}
        self.det_config = self.config_dict['table_detection']['config_path']
        self.det_ckpt = self.config_dict['table_detection']['model_path']
        self.det_thr = self.config_dict['table_detection']['score_thr']
        if self._DETECTION_MODEL is None:
            with self.timer("Load table detection model"):
                self._DETECTION_MODEL = init_detector(self.det_config, self.det_ckpt, device='cuda:0')
        if self._LINE_MODEL is None:
            with self.timer("Load line segmentation model"):
                self._LINE_MODEL = table_net((*self.config_dict['table_line']['input_shape'], 3), num_classes=2)
                self._LINE_MODEL.load_weights(self.config_dict['table_line']['model_path'])
        if self._OCR_MODEL is None:
            det_model_dir = self.config_dict['ocr'][self.config_dict['ocr']['type']]['det_path']
            rec_model_dir = self.config_dict['ocr'][self.config_dict['ocr']['type']]['rec_path']
            self._OCR_MODEL = self._OCR_FACTORY[self.config_dict['ocr']['type']](det_model_dir, rec_model_dir)
        if preload:
            with self.timer('Preload LINE MODEL'):
                inputBlob = np.random.randn(*self.config_dict['table_line']['input_shape'], 3)
                _ = self._LINE_MODEL.predict(np.array([np.array(inputBlob) / 255.0]))

    # noinspection PyUnresolvedReferences
    def _configure_gpu(self):
        if self.device.startswith('cuda') and torch.version.cuda.startswith('11'):
            # tensorflow compatibility settings for CUDA11, other CUDA environments will skip code below
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession
            config = ConfigProto()
            # config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)

    def pipeline(self, ori_img: np.ndarray, **kwargs):
        # this pipeline has to handle following options:
        # a. perspective transformation enable or not (clear paper edge must be seen if enable)
        # b. table detection enable or not (whether there are multiple tables in the same page)
        # c. table cell extraction & OCR & match are compulsory
        p_trans = kwargs['p_trans']
        t_detection = kwargs['t_detection']
        ocr_det_disable = kwargs['ocr_det_disable']
        async_cell_ocr = kwargs['async_cell_ocr']
        ret_stages = {
            'debug': ''
        }

        # 0. preprocessing
        with self.timer("preprocessing") as t_preprocess:
            preprocessed_img, flagged = self._preprocess(ori_img, p_trans=p_trans)

        # 1. table detection for boarded & boarderless table
        with self.timer("table detection") as t_det:
            if t_detection:
                tables = self._get_tables(preprocessed_img)
            else:
                # if disable table detection, we treat the whole image as a table
                tables = np.array([[0, 0, *preprocessed_img.shape[1::-1]]])
            # 2. get cells using UNet and postprocessing
        with self.timer("cells extraction") as t_cell:
            cells = self._get_cells(preprocessed_img, tables)  # cells coords not changed

        # 3. get content from OCR (can be async between cell extraction if performance not fit)
        if async_cell_ocr and ocr_det_disable:
            raise ValueError("Async processing & ocr_det_disable can't be turned on in the same time!")
        with self.timer("OCR pipeline") as t_ocr:
            dt_boxes = None
            if ocr_det_disable:
                if len(cells) == 1:
                    ret_stages['debug'] += 'ocr_det_disable'
                    dt_boxes = cells[0].reshape((-1, 4, 2)).astype(np.int).astype(np.float32)
                else:
                    ret_stages['debug'] += 'ocr_det_enable_since_more_than_one_table_detected'
            ocr = self._get_ocr(preprocessed_img, dt_boxes=dt_boxes)  # only

        # 4. match OCR content with cells
        with self.timer("Match OCR with cells") as t_match:
            self._match(cells, ocr, tables)

        # 5. save all middle results, ready to display on webpage

        req_id = str(uuid.uuid4())
        tmp_cell = preprocessed_img.copy()
        # for i in range(len(cells)):  # TODO: draw self.tables
        #     tmp_cell = draw_boxes(tmp_cell, cells[i].tolist())
        for j in range(len(self.tables)):
            for cell in self.tables[j].table_cells:
                tmp_cell = draw_boxes(tmp_cell, [cell.coord])

        ret_stages.update(
            total_time=t_preprocess.msecs + t_det.msecs + t_cell.msecs + t_ocr.msecs + t_match.msecs,
            original=os.path.join(self.static_prefix, req_id + '_original.jpg'),
            cell=[t_cell.msecs, os.path.join(self.static_prefix, req_id + '_cell.jpg')]
        )

        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_original.jpg'), ori_img)
        if p_trans:
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_pre.jpg'), preprocessed_img)
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_flagged.jpg'), flagged)
            ret_stages.update(
                preprocessing=[t_preprocess.msecs, os.path.join(self.static_prefix, req_id + '_pre.jpg')],
                flagged=os.path.join(self.static_prefix, req_id + '_flagged.jpg')
            )
        if t_detection:
            table_img = draw_boxes(preprocessed_img, tables)
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_table.jpg'), table_img)
            ret_stages.update(
                det=[t_det.msecs, os.path.join(self.static_prefix, req_id + '_pre.jpg')]
            )
        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_cell.jpg'), tmp_cell)

        return ret_stages

    def _preprocess(self, ori_img: np.ndarray, p_trans=False) -> (np.ndarray, np.ndarray):
        """
        Apply four-point perspective transformation based on the largest rectangle
        This requires clear edges can be seen in original image (paper edge or single table edge)
        :param ori_img: np.ndarray
        :param p_trans: bool
        :return: wrapped, drawn
        """
        if not p_trans:
            return ori_img, ori_img
        return preprocess(ori_img)

    def _get_tables(self, ori_img) -> np.ndarray:  # xyxy
        h, w, _ = ori_img.shape
        coords, _ = inference_detector(self._DETECTION_MODEL, ori_img)
        coords: List[np.ndarray]
        boarded, cell, boarderless = coords[:3]  # only extract relevant 3 types
        # boarded example: [[ 13.042118    18.262554   621.037      244.12543      0.90694094], [...], ...]  x,y,x,y,conf
        boarded_ids: np.ndarray = np.where(boarded[:, -1] > self.det_thr)[0]
        x1, y1, x2, y2, conf = np.split(boarded[boarded_ids], 5, axis=1)
        # adjust bbox according to confidence
        pad_x = (x2 - x1) * (1 - conf)  # higher conf leads to lower padding
        pad_y = (y2 - y1) * (1 - conf)
        x1 = x1 - pad_x
        y1 = y1 - pad_y
        x2 = x2 + pad_x
        y2 = y2 + pad_y
        # set boundaries for adjusted bbox
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        x2[x2 > w] = w
        y2[y2 > h] = h

        return np.hstack((x1, y1, x2, y2))
        # return np.array([[0, 0, *ori_img.shape[1::-1]]])

    def _get_ocr(self, ori_img, dt_boxes=None) -> dict:  # in paddle
        return self._OCR_MODEL.get_result(ori_img, dt_boxes=dt_boxes)

    def _get_cells(self, ori_img, table_coords) -> List[np.ndarray]:  # in tensorflow
        cells = []
        # table_imgs = []
        for coord in table_coords:  # for each boarded table
            xmin, ymin, xmax, ymax = [int(k) for k in coord]  # used for cropping & shifting
            table_img = ori_img[ymin:ymax, xmin:xmax]  # cropped img
            # row_boxes, col_boxes = table_line(self._LINE_MODEL, table_img[..., ::-1],  # BGR->RGB
            #                                       size=self.unet_shape,
            #                                       hprob=self.hprob,
            #                                       vprob=self.vprob)
            with Timer("lines extraction(in cells extraction)"):
                row_boxes, col_boxes = table_line(self._LINE_MODEL, table_img[..., ::-1],
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
        assert len(cells) == len(tables), "table & cell not match!"
        self.tables = []
        for i in range(len(tables)):
            my_table = Table(coord=tables[i], cells=cells[i], verbose=False)
            my_table.match_ocr(ocr)
            my_table.build_structure()
            self.tables.append(my_table)

    def to_excel(self, filename='./debug.xlsx'):
        workbook = xlsxwriter.Workbook(filename)
        for (i, table) in enumerate(self.tables):
            name = table.title if table.title != '' else '识别表格' + str(i)
            worksheet = workbook.add_worksheet(name=name)
            for (row_num, row) in enumerate(table.rows):
                row: List[TableCell]
                for cell in row:
                    if cell.col_range[0] != cell.col_range[1]:
                        worksheet.merge_range(row_num, cell.col_range[0], row_num, cell.col_range[1],
                                              data='\n'.join(cell.ocr_content))
                    else:
                        worksheet.write(row_num, cell.col_range[0], '\n'.join(cell.ocr_content))
        workbook.close()


if __name__ == '__main__':
    # unittests for WebHandler
    handler = WebHandler()
    ori_img = cv2.imread('./test_720p_2.JPG')
    print(handler.pipeline(ori_img,
                           p_trans=True,
                           t_detection=False,
                           ocr_det_disable=False,
                           async_cell_ocr=False))
    handler.to_excel()
