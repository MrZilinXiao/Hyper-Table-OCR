import pytest
import json
import cv2

from table import Table
from web import WebHandler


def test_pipeline():
    handler = WebHandler(debug=True)
    ori_img = cv2.imread('merged.jpg')
    # test pipeline except OCR
    # handler.pipeline(ori_img)
    tables = handler._get_tables(ori_img)
    cells = handler._get_cells(ori_img, tables)
    ocr = json.load(open('test_merged.json', 'r'))
    handler._match(cells, ocr, tables)
    print('test done')

# def test_table_class():
#     coord = [0, 0, 100, 100]  # useless for now
#     ori_img = cv2.imread('merged.jpg')
#     # table = Table(coord)


if __name__ == '__main__':
    pytest.main()
