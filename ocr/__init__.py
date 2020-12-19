import copy

from PIL import Image

from chineseocr.crnn import FullCrnn, CRNNHandle
from chineseocr.dbnet.dbnet_infer import DBNET
from chineseocr.crnn.keys import alphabetChinese as alphabet
# from chineseocr.crnn.CRNN import CRNNHandle

import abc
from ocr.tools.infer.predict_system import TextSystem
from utils import Timer, sorted_boxes, get_rotate_crop_image
import argparse
import numpy as np


class OCRHandler(metaclass=abc.ABCMeta):
    """
    Handler for OCR Support
    An abstract class, any OCR implementations may derive from it
    """

    def __init__(self, *kw, **kwargs):
        pass

    def get_result(self, ori_img):
        """
        Interface for OCR inference
        :param ori_img: np.ndarray
        :return: dict, in following format:
        {'sentences': [['麦格尔特杯表格OCR测试表格2', [[85.0, 10.0], [573.0, 30.0], [572.0, 54.0], [84.0, 33.0]], 0.9],...]}
        """
        pass


class PaddleHandler(OCRHandler):
    use_angle_cls = False
    use_space_char = True
    gpu_mem = 4096

    def __init__(self, det_model_dir="./models/ch_ppocr_server_v1.1_det_infer/",
                 rec_model_dir="./models/ch_ppocr_mobile_v1.1_rec_infer/",
                 cls_model_dir=""):
        super(PaddleHandler, self).__init__()
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        args = self._load_parser()
        self.model = TextSystem(args)
        # warmup to maximum inference speed
        self._warmup()

    def _load_parser(self):
        parser = argparse.ArgumentParser()
        # params for prediction engine
        parser.add_argument("--use_gpu", type=bool, default=True)
        parser.add_argument("--ir_optim", type=bool, default=True)
        parser.add_argument("--use_tensorrt", type=bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str)
        parser.add_argument("--det_max_side_len", type=float, default=960)

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # SAST parmas
        parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
        parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
        parser.add_argument("--det_sast_polygon", type=bool, default=False)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument("--rec_char_dict_path", type=str, default="./ppocr/utils/ppocr_keys_v1.txt")
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--vis_font_path", type=str, default="./doc/simfang.ttf")

        # params for text classifier
        parser.add_argument("--use_angle_cls", type=bool, default=False)
        parser.add_argument("--cls_model_dir", type=str)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=30)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)

        parser.add_argument("--use_pdserving", type=bool, default=False)

        # 是否可视化
        parser.add_argument("--is_visualize", type=bool, default=True)

        args = parser.parse_args()
        args.det_model_dir = self.det_model_dir
        args.rec_model_dir = self.rec_model_dir
        args.cls_model_dir = self.cls_model_dir
        args.use_angle_cls = self.use_angle_cls
        args.use_space_char = self.use_space_char
        args.gpu_mem = self.gpu_mem
        return args

    def _warmup(self):
        with Timer("OCR Warmup"):
            tmp = np.random.randn(224, 224, 3)
            self.model(tmp)

    def get_result(self, ori_img, rec_thr=0.5, dt_boxes=None):
        bbox_list, text_list, score_list = [], [], []
        dt_boxes, rec_res = self.model(ori_img, dt_boxes=dt_boxes)
        for j, (text, score) in enumerate(rec_res):
            if score >= rec_thr:
                bbox_list.append(dt_boxes[j])
                text_list.append(text)
                score_list.append(score)
        sentences = []
        for k in range(len(bbox_list)):
            sentence = [text_list[k], bbox_list[k].tolist(), score_list[k]]
            sentences.append(sentence)
        return {'sentences': sentences}


class ChineseOCRHandler(OCRHandler):
    """
    Handler for OCR support
    Word segmentation: dbnet, psenet
    Recognition: crnn
    """
    _det_factory = {
        'dbnet': DBNET
    }
    _rec_factory = {
        'full_lstm': FullCrnn
    }

    def __init__(self, det_type='dbnet',
                 rec_type='full_lstm',
                 det_model_path='./models/dbnet.onnx',
                 dbnet_short_size=960,
                 rec_model_path='./models/ocr-lstm.pth',
                 device='cuda:0'):
        super(OCRHandler, self).__init__()
        if det_type not in self._det_factory:
            raise ValueError("OCR detection model '{}' not support!".format(det_type))
        if rec_type not in self._rec_factory:
            raise ValueError("OCR recognition model '{}' not support!".format(rec_type))
        self.det_net = self._det_factory[det_type](det_model_path, short_size=dbnet_short_size)
        self.rec_net = self._rec_factory[rec_type](32, 1, len(alphabet) + 1, 256, n_rnn=2, leakyRelu=False,
                                                   lstmFlag='lstm' in rec_type)
        self.rec_handle = CRNNHandle(rec_model_path, self.rec_net, device=device)
        # self._warmup()

    def _warmup(self):
        with Timer("ONNX warmup"):
            tmp = np.random.randn(1024, 1024, 3).astype(np.uint8)
            self.det_net.process(tmp)

    def crnnRecWithBox(self, im, boxes_list):
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))
        for index, box in enumerate(boxes_list):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))

            partImg = Image.fromarray(partImg_array).convert("RGB")
            partImg_ = partImg.convert('L')
            newW, newH = partImg.size
            try:
                simPred = self.rec_handle.predict(partImg_)  ##识别的文本
            except:
                continue
            if simPred.strip() != u'':
                # results.append({'cx': 0, 'cy': 0, 'text': simPred, 'w': newW, 'h': newH,
                # 'degree': 0})
                results.append(simPred)
                # results.append({ 'text': simPred, })
        return results

    @staticmethod
    def _convert(result, boxes_list, scores_list):
        sentences = {'sentences': []}
        for i, _str in enumerate(result):
            box = boxes_list[i].tolist()
            score = round(scores_list[i].astype(np.float32), 2)
            sentences['sentences'].append([_str, box, score])
        return sentences

    def get_result(self, img: np.ndarray, dt_boxes=None):
        #  {'sentences': [['麦格尔特杯表格OCR测试表格2', [[85.0, 10.0], [573.0, 30.0], [572.0, 54.0], [84.0, 33.0]], 0.9],...]}
        with Timer('ONNX dbnet'):
            boxes_list, scores_list = self.det_net.process(img)  # onnxruntime can be run on GPU
        boxes_list = sorted_boxes(boxes_list)
        # boxes_list: List[np.ndarray] (N, 4, 2)
        with Timer('CRNN'):
            result = self.crnnRecWithBox(img, boxes_list)
        return self._convert(result, boxes_list, scores_list)


if __name__ == '__main__':
    import cv2

    # unittests for PaddleOCR
    handler = PaddleHandler()
    img = cv2.imread('./merged.jpg')
    print(handler.get_result(img))

    chinese_handler = ChineseOCRHandler()
    print(chinese_handler.get_result(img))
    pass

    # unittests for ChineseOCR

