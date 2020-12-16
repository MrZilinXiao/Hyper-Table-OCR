from abc import ABC
from flask import Flask, request, jsonify, views, render_template
import gevent
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
# from geventwebsocket.websocket import WebSocket
from utils import MyLogger, RemoteLogger
from web import WebHandler
import os
import yaml
import time

ALLOWED_EXTENSIONS = {'png', 'jpg', 'bmp'}


def is_allow_extension(filename: str):
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


class MainView(views.View, ABC):
    methods = ['GET']
    config_dict = yaml.load(open('./config.yml', 'r'), Loader=yaml.Loader)
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
        'p_trans': ('启用透视变换', True)
    }

    # _online_log_path = 'web_remote.log'

    def __init__(self):
        super(MainView, self).__init__()
        self.ws_logger = RemoteLogger(debug=True)

    def dispatch_request(self):
        ocr_method_list = [{'value': k, 'desp': '%d: ' % idx + self._ocr_mapping[k]} for idx, k in
                           enumerate(self.config_dict['ocr'].keys()) if k != 'type']
        cell_method_list = [{'value': k, 'desp': '%d: ' % idx + self._cell_mapping[k]} for idx, k in
                            enumerate(self.config_dict['table_line'].keys()) if k != 'type']
        advance_options = [{'value': k, 'desp': self._advance_options[k][0], 'enable': self._advance_options[k][1]} for
                           k in self._advance_options.keys()]
        return render_template('index.html', ocr=ocr_method_list, cell=cell_method_list, advance=advance_options)


class UploadView(views.View, ABC):
    methods = ['POST']

    def __init__(self):
        super(UploadView, self).__init__()

    def dispatch_request(self):
        form = dict(request.form)
        f = request.files['local_image']
        if not is_allow_extension(f.filename):
            return jsonify({'status': '20001', 'desp': '文件格式暂不支持'})
        submit_kwargs = {}
        for k in form.keys():
            if form[k] == '-1':
                return jsonify({'status': '30001', 'desp': '%s 未选中有效选项！' % k})
            elif form[k] == 'on':
                submit_kwargs[k] = True
            elif form[k] == 'off':
                submit_kwargs[k] = False
            else:
                submit_kwargs[k] = form[k]

        return jsonify()


global app
app = Flask(__name__, static_url_path='',
            static_folder='./static',
            template_folder='./templates')
main_view = MainView()
upload_view = UploadView()
app.add_url_rule('/', view_func=main_view.as_view('index'))
app.add_url_rule('/upload', view_func=upload_view.as_view('upload'))


@app.route('/log')
def log():
    user_socket = request.environ.get("wsgi.websocket")
    logger = main_view.ws_logger
    logger.info("Remote Log Websocket established: " + str(user_socket) + request.remote_addr)
    while True:
        time.sleep(1)
        send_ret = ''.join(logger.lines[::-1])
        user_socket.send(send_ret)


if __name__ == '__main__':
    # http_serv = WSGIServer((main_view.config_dict['web']['host'], main_view.config_dict['web']['port']), app, handler_class=WebSocketHandler)
    # http_serv.serve_forever()
    app.run(debug=True, threaded=True, **main_view.config_dict['web'])
