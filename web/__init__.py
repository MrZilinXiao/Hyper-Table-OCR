import tensorflow as tf

from PIL import Image

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

import xlsxwriter

from utils import Singleton, Timer, EmptyTimer, draw_boxes, RemoteLogger, eval_angle
from typing import List, Union

from boardered.table_net import table_net
from boardered.extractor import UNetExtractor, TraditionalExtractor
from table import Table, TableCell
from table.transform import preprocess
from table.HED import HEDNet
from ocr import PaddleHandler, ChineseOCRHandler
from ocr.tools.infer.utility import draw_ocr_box_txt
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
    _OCR_MODEL_LITE = None
    _OCR_CHINESE_OCR = None
    _OCR_FACTORY = {
        'paddle': PaddleHandler,
        'paddle_lite': PaddleHandler,
        'chineseocr': ChineseOCRHandler
    }

    def __init__(self, config: Union[str, dict] = './config.yml', debug=True, preload=True, device='cuda:0',
                 static_folder='static/', local_no_gpu_test=False):
        if local_no_gpu_test:  # for local test, will not loading any models!
            return
        self.device = device
        if not tf.test.is_gpu_available():
            self.device = 'cpu'
        self.static_prefix = static_folder
        if not os.path.exists(static_folder):
            os.mkdir(static_folder)
        self.debug = debug
        self.timer = Timer if self.debug else EmptyTimer
        self.config_dict: dict = yaml.load(open(config, 'r'), Loader=yaml.Loader) if isinstance(config, str) else config
        self.hprob, self.vprob = self.config_dict['table_line']['unet']['hprob'], \
                                 self.config_dict['table_line']['unet']['vprob']
        self.unet_shape = self.config_dict['table_line']['unet']['input_shape']
        # {'table_line': {'model_path': './table-line.h5', 'input_shape': [512, 512]}, 'ocr': {'model_path': ''}, 'web': {'host': '0.0.0.0', 'port': 23333}}
        self.det_config = self.config_dict['table_detection']['config_path']
        self.det_ckpt = self.config_dict['table_detection']['model_path']
        self.det_thr = self.config_dict['table_detection']['score_thr']
        if self._DETECTION_MODEL is None:
            with self.timer("Load table detection model"):
                self._DETECTION_MODEL = init_detector(self.det_config, self.det_ckpt, device='cuda:0')
        if self._LINE_MODEL is None:
            with self.timer("Load line segmentation model"):
                self._LINE_MODEL = table_net((*self.config_dict['table_line']['unet']['input_shape'], 3), num_classes=2)
                self._LINE_MODEL.load_weights(self.config_dict['table_line']['unet']['model_path'])
                self._UNET_CELL_HANDLER = UNetExtractor(self._LINE_MODEL, self.unet_shape, self.hprob, self.vprob)

        self._TRADITIONAL_CELL_HANDLER = TraditionalExtractor()

        if self._OCR_MODEL is None:
            det_model_dir = self.config_dict['ocr']['paddle']['det_path']
            rec_model_dir = self.config_dict['ocr']['paddle']['rec_path']
            self._OCR_MODEL = self._OCR_FACTORY['paddle'](det_model_dir, rec_model_dir)
        if self._OCR_MODEL_LITE is None:
            det_model_dir = self.config_dict['ocr']['paddle_lite']['det_path']
            rec_model_dir = self.config_dict['ocr']['paddle_lite']['rec_path']
            self._OCR_MODEL_LITE = self._OCR_FACTORY['paddle_lite'](det_model_dir, rec_model_dir)

        if self._OCR_CHINESE_OCR is None:
            det_model_dir = self.config_dict['ocr']['chineseocr']['det_path']
            rec_model_dir = self.config_dict['ocr']['chineseocr']['rec_path']
            self._OCR_CHINESE_OCR = self._OCR_FACTORY['chineseocr'](det_model_path=det_model_dir, rec_model_path=rec_model_dir)

        self.hed_model = HEDNet(ckpt_path=self.config_dict['preprocessing']['hed_path']).to(self.device)
        # self.hed_model.load_weight()

        self.ocr_options = {
            'paddle': self._OCR_MODEL,
            'paddle_lite': self._OCR_MODEL_LITE,
            'chineseocr': self._OCR_CHINESE_OCR
        }
        if preload:
            with self.timer('Preload LINE MODEL'):
                inputBlob = np.random.randn(*self.config_dict['table_line']['unet']['input_shape'], 3)
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
        p_trans_options = kwargs['p_trans_options'] if 'p_trans_options' in kwargs else None

        t_detection = kwargs['t_detection']
        ocr_det_disable = kwargs['ocr_det_disable']
        ocr_type = kwargs['ocr']
        adjust_angle = kwargs['adjust_angle']
        traditional_cell = kwargs['cell'] == 'traditional'
        # async_cell_ocr = kwargs['async_cell_ocr']
        ret_stages = {
            'debug': ''
        }

        # 0. preprocessing
        with self.timer("preprocessing") as t_preprocess:
            preprocessed_img, flagged = self._preprocess(ori_img, p_trans=p_trans, adjust_angle=adjust_angle, p_trans_options=p_trans_options)

        # 1. table detection for boarded & boarderless table
        with self.timer("table detection") as t_det:
            if t_detection:
                tables = self._get_tables(preprocessed_img)
            else:
                # if disable table detection, we treat the whole image as a table
                tables = np.array([[0, 0, *preprocessed_img.shape[1::-1]]])
            # 2. get cells using UNet and postprocessing
        with self.timer("cells extraction") as t_cell:
            cells = self._get_cells(preprocessed_img, tables, traditional=traditional_cell)  # cells coords not changed

        # 3. get content from OCR (can be async between cell extraction if performance not fit)
        # TODO: consider using dual GPUs to perform async operations
        # if async_cell_ocr and ocr_det_disable:
        #     raise ValueError("Async processing & ocr_det_disable can't be turned on in the same time!")
        with self.timer("OCR pipeline") as t_ocr:
            dt_boxes = None
            if ocr_det_disable:
                if len(cells) == 1:
                    ret_stages['debug'] += 'ocr_det_disable'
                    dt_boxes = cells[0].reshape((-1, 4, 2)).astype(np.int).astype(np.float32)
                else:
                    ret_stages['debug'] += 'ocr_det_enable_since_more_than_one_table_detected'
            ocr = self._get_ocr(preprocessed_img, _type=ocr_type, dt_boxes=dt_boxes)

        # 4. match OCR content with cells
        with self.timer("Match OCR with cells") as t_match:
            match_img = self._match(cells, ocr, tables, preprocessed_img)

        # 5. save all middle results, ready to display on webpage

        req_id = str(uuid.uuid4())
        tmp_cell = preprocessed_img.copy()
        # for i in range(len(cells)):  # Done: draw self.tables
        #     tmp_cell = draw_boxes(tmp_cell, cells[i].tolist())
        for j in range(len(self.tables)):
            for row in self.tables[j].rows:
                for cell in row:
                    tmp_cell = draw_boxes(tmp_cell, [cell.coord])

        ret_stages.update(
            total_time=t_preprocess.msecs + t_det.msecs + t_cell.msecs + t_ocr.msecs + t_match.msecs,
            original=req_id + '_original.jpg',
            cell=[t_cell.msecs, req_id + '_cell.jpg'],
            ocr=[t_ocr.msecs, req_id + '_ocr.jpg'],
            match=[t_match.msecs, req_id + '_match.jpg']
        )

        # cv2.imshow('pure match', match_img)
        # cv2.waitKey(0)

        # visualize OCR result
        image = Image.fromarray(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))

        # boxes = dt_boxes  # [-1, 4, 2]
        boxes, txts, scores = [], [], []
        ocr['sentences']: List[List[str, List[List[int, int]], float]]
        for txt, coord, score in ocr['sentences']:
            txts.append(txt)
            scores.append(score)
            boxes.append(coord)
        boxes = np.array(boxes)

        img_left, img_right = draw_ocr_box_txt(  # TODO: see img_left, why img_left is exactly the same with match_img
            image,
            boxes,
            txts,
            scores,
            drop_score=0.5,
            font_path='./simfang.ttf')
        # left is
        # h, w = img_left.
        # img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        # img_show.paste(img_left, (0, 0, w, h))
        # img_show.paste(img_right, (w, 0, w * 2, h))
        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_ocr.jpg'),
                    img_right[:, :, ::-1])

        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_match.jpg'),
                    img_left[:, :, ::-1])

        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_original.jpg'), ori_img)
        if p_trans or adjust_angle:
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_pre.jpg'), preprocessed_img)
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_flagged.jpg'), flagged)
            ret_stages.update(
                preprocessing=[t_preprocess.msecs, req_id + '_pre.jpg'],
                flagged=req_id + '_flagged.jpg'
            )
        if t_detection:
            table_img = draw_boxes(preprocessed_img, tables)
            cv2.imwrite(os.path.join(self.static_prefix, req_id + '_table.jpg'), table_img)
            ret_stages.update(
                det=[t_det.msecs, os.path.join(req_id + '_table.jpg')]
            )
        cv2.imwrite(os.path.join(self.static_prefix, req_id + '_cell.jpg'), tmp_cell)
        self.to_excel(filename=os.path.join(self.static_prefix, req_id + '.xlsx'), need_title=(not t_detection) or len(tables) == 1)
        ret_stages.update(excel=req_id + '.xlsx')

        return ret_stages

    def _preprocess(self, ori_img: np.ndarray, p_trans=False, adjust_angle=False, p_trans_options='traditional') -> (np.ndarray, np.ndarray):
        """
        Apply four-point perspective transformation based on the largest rectangle
        This requires clear edges can be seen in original image (paper edge or single table edge)
        :param ori_img: np.ndarray
        :param p_trans: bool
        :return: wrapped, drawn
        """
        flagged = None
        if p_trans:
            ori_img, flagged = preprocess(ori_img, hed_model=self.hed_model if p_trans_options == 'hed' else None)
        if adjust_angle:
            ori_img, degree = eval_angle(ori_img, [-5, 5])
            RemoteLogger.info("开启了角度校正，度数为%d°" % degree)
        return ori_img, flagged if flagged is not None else ori_img

    def _get_cells(self, ori_img, tables, traditional=False):
        return self._TRADITIONAL_CELL_HANDLER.get_cells(ori_img,
                                                        tables) if traditional else self._UNET_CELL_HANDLER.get_cells(
            ori_img, tables)

    def _get_tables(self, ori_img, top_left_pad_ratio=0.1) -> np.ndarray:  # xyxy
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
        x1 = x1 - pad_x - (x2 - x1) * top_left_pad_ratio
        y1 = y1 - pad_y - (y2 - y1) * top_left_pad_ratio
        x2 = x2 + pad_x
        y2 = y2 + pad_y
        # set boundaries for adjusted bbox
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        x2[x2 > w] = w
        y2[y2 > h] = h

        return np.hstack((x1, y1, x2, y2))
        # return np.array([[0, 0, *ori_img.shape[1::-1]]])

    def _get_ocr(self, ori_img, _type='paddle', dt_boxes=None) -> dict:
        try:
            return self.ocr_options[_type].get_result(ori_img, dt_boxes=dt_boxes)
        except AttributeError:  # selected OCR model is not loaded
            raise RuntimeError("选择的模型%s未被加载，请修改配置！" % _type)

    def _match(self, cells, ocr: dict, tables, img):
        if len(cells) != len(tables) or cells is None:
            raise RuntimeError("对表格未检出Cell！")
        self.tables = []
        for i in range(len(tables)):
            my_table = Table(coord=tables[i], cells=cells[i], verbose=False)
            img = my_table.match_ocr(ocr, img)
            my_table.build_structure()
            self.tables.append(my_table)
        return img

    def to_excel(self, filename='./debug.xlsx', need_title=False):
        workbook = xlsxwriter.Workbook(filename)
        for (i, table) in enumerate(self.tables):
            name = table.title[:31] if need_title else '识别表格' + str(i)  # clip to 31 chars to avoid Excel worksheet name exception
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
                           ocr_det_disable=False
                           ))
    # handler.to_excel()
