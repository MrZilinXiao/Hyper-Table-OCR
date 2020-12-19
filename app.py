from abc import ABC

# import gevent
import gevent.monkey
from flask import Flask, request, jsonify, views, render_template

gevent.monkey.patch_all()
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
from utils import RemoteLogger
from web import WebHandler
import os
import yaml
import cv2
import time

ALLOWED_EXTENSIONS = {'png', 'jpg', 'bmp'}


def is_allow_extension(filename: str):
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, static_url_path='',
            static_folder='./static',
            template_folder='./templates')


class MainView(views.View, ABC):
    methods = ['GET']
    config_dict = yaml.load(open('./config.yml', 'r'), Loader=yaml.Loader)
    # web_handler = WebHandler(config_dict, static_folder=app.static_folder, local_no_gpu_test=True)
    web_handler = WebHandler(config_dict, static_folder=app.static_folder)
    _cell_mapping = {
        'unet': 'UNet网格线分割',
        'traditional': '传统方法分割'
    }
    _ocr_mapping = {
        'paddle': 'Paddle推理',
        'paddle_lite': 'PaddleLite推理',
        'chineseocr': 'ChineseOCR推理'
    }
    _advance_options = {
        't_detection': ('启用表格检测', False),
        'ocr_det_disable': ('禁用OCR定位模块', False),
        'p_trans': ('启用透视变换', True),
        'adjust_angle': ('启用角度纠正', False)
    }
    _p_trans_options = {
        'traditional': 'Canny算子',
        'hed': 'HED边缘检测'
    }

    # _online_log_path = 'web_remote.log'

    def __init__(self):
        super(MainView, self).__init__()

    def dispatch_request(self):
        ocr_method_list = [{'value': k, 'desp': '%d: ' % idx + self._ocr_mapping[k]} for idx, k in
                           enumerate(self.config_dict['ocr'].keys()) if k != 'type']
        cell_method_list = [{'value': k, 'desp': '%d: ' % idx + self._cell_mapping[k]} for idx, k in
                            enumerate(self.config_dict['table_line'].keys()) if k != 'type']
        advance_options = [{'value': k, 'desp': self._advance_options[k][0], 'enable': self._advance_options[k][1]} for
                           k in self._advance_options.keys()]
        p_trans_options = [{'value': k, 'desp': '%d: ' % idx + self._p_trans_options[k]}
                           for idx, k in enumerate(self._p_trans_options.keys())]
        return render_template('index.html',
                               ocr=ocr_method_list,
                               cell=cell_method_list,
                               advance=advance_options,
                               p_trans_options=p_trans_options)

    @property
    def advance_options(self):
        return self._advance_options


class UploadView(views.View, ABC):
    methods = ['POST']
    tmp_root = './tmp'

    def __init__(self):
        super(UploadView, self).__init__()
        if not os.path.exists(self.tmp_root):
            os.mkdir(self.tmp_root)

    def dispatch_request(self):
        try:
            form = dict(request.form)
            f = request.files['local_image']
            if not is_allow_extension(f.filename):
                return jsonify({'status': '20001', 'desp': '文件格式暂不支持'})
            submit_kwargs = {k: False for k in main_view.advance_options}
            for k in form.keys():
                if form[k] == '-1':
                    return jsonify({'status': '30001', 'desp': '%s 未选中有效选项！' % k})
                elif form[k] == 'on':
                    submit_kwargs[k] = True
                else:
                    submit_kwargs[k] = form[k]

            # submit_kwargs: {'p_trans': True, 't_detection': False, 'ocr_det_disable': False}
            # write img to tmp file
            f.save(os.path.join(self.tmp_root, f.filename))
            ori_img = cv2.imread(os.path.join(self.tmp_root, f.filename))
            ret_details = main_view.web_handler.pipeline(ori_img, **submit_kwargs)
            RemoteLogger.info("完成！调试信息：%s" % ret_details)
            # handler return info: {'debug': '', 'total_time': 370.9299564361572,
            # 'original': '1c9e9711-f01d-4113-b11e-e546428a759a_original.jpg',
            # 'cell': [111.13262176513672, '1c9e9711-f01d-4113-b11e-e546428a759a_cell.jpg'],
            # 'ocr': [239.74919319152832, '1c9e9711-f01d-4113-b11e-e546428a759a_ocr.jpg'],
            # 'preprocessing': [2.897500991821289, '1c9e9711-f01d-4113-b11e-e546428a759a_pre.jpg'],
            # 'flagged': '1c9e9711-f01d-4113-b11e-e546428a759a_flagged.jpg',
            # 'excel': '1c9e9711-f01d-4113-b11e-e546428a759a.xlsx'}
            return jsonify({'status': '10000', 'details': ret_details})
        except Exception as e:  # for friendly prompts
            RemoteLogger.info(str(e))
            return jsonify({'status': '40001', 'desp': '未预见的错误: %s' % str(e)})


main_view = MainView()
upload_view = UploadView()
app.add_url_rule('/', view_func=main_view.as_view('index'))
app.add_url_rule('/upload', view_func=upload_view.as_view('upload'))


@app.route('/log')
def log():
    user_socket = request.environ.get("wsgi.websocket")
    logger = RemoteLogger
    logger.info("Remote Log Websocket established: " + str(user_socket) + request.remote_addr)
    while True:
        time.sleep(1)
        send_ret = ''.join(logger.lines[::-1])
        user_socket.send(send_ret)


if __name__ == '__main__':
    http_serv = WSGIServer((main_view.config_dict['web']['host'], main_view.config_dict['web']['port']), app,
                           handler_class=WebSocketHandler)
    RemoteLogger.info("服务启动完成！")
    http_serv.serve_forever()
    # app.run(debug=False, **main_view.config_dict['web'])
