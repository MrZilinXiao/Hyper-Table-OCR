# Hyper-Table-OCR

A carefully-designed OCR pipeline for universal boarded table recognition and reconstruction.

This pipeline covers image preprocessing, table detection(optional), text OCR, table cell extraction, table reconstruction.

Are you seeking ideas for your own work? Visit [my blog post on Hyper-Table-OCR](https://mrxiao.net/hyper-table-ocr.html) to see more!

**Update on 2021-08-20: Happy to see that Baidu has released their [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/ppstructure/README.md), which provides higher robustness due to its DL-driven structure prediction feature, instead of simple matching in our work.**

## Demo

![gif demo](https://upyun.mrxiao.net/img/demo.gif)

Demo Video (In English): [YouTube](https://youtu.be/v2pe6cAofcw)

[![Hyper Table Recognition: A carefully-designed Table OCR pipeline](https://res.cloudinary.com/marcomontalbano/image/upload/v1608561697/video_to_markdown/images/youtube--v2pe6cAofcw-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/v2pe6cAofcw "Hyper Table Recognition: A carefully-designed Table OCR pipeline")

Demo Video (In Chinese): [Bilibili](https://www.bilibili.com/video/BV1K64y1Z7XB)

## Features

- Flexible modular architecture: by deriving from predefined abstract class, any module of this pipeline can be easily swapped to your preferred one. *See the following `Want to contribute?` part!*
- A simple yet highly legible web interface.
- A table reconstruction strategy based simply on coordinates of each cell, including identifying merged cell row & building table structure.
- More to explore...

## Getting Started

### Clone this repo

```bash
git clone https://github.com/MrZilinXiao/Hyper-Table-Recognition
cd Hyper-Table-Recognition
```

### Download weights

Download from here: [GoogleDrive](https://drive.google.com/file/d/10NynXURJP2y1M7ScB0lzXnWcXba30PhM/view?usp=sharing) 

MD5: (004fabb8f6112d6d43457c681b435631  models.zip)

Unzip it and make sure the directory layout matchs:

```bash
# ~/Hyper-Table-Recognition$ tree -L 1
.
├── models
├── app.py
├── config.yml
├── ...
```

### Install Dependencies

This project is developed and tested on:

- Ubuntu 18.04
- RTX 3070 with Driver 455.45.01 & CUDA 11.1 & cuDNN 8.0.4
- Python 3.8.3
- PyTorch 1.7.0+cu110
- Tensorflow 2.5.0
- PaddlePaddle 2.0.0-rc1
- mmdetection 2.7.0
- onnxruntime-gpu 1.6.0

An NVIDIA GPU device is compulsory for reasonable inference duration, while GPU with less than 6GB VRAM may experience `Out of Memory` exception when loading multiple models. You may comment some models in `web/__init__.py` if experiencing such situation.

> No version-specific framework feature is used in this project, so this means you could still enjoy it with lower versions of these frameworks. However, at this time(19th Dec, 2020), users with RTX 3000 Series device may have no access to compiled binary of Tensorflow, onnxruntime-gpu, mmdetection, PaddlePaddle via `pip` or `conda`.
>
> Some building tutorials for Ubuntu are as follows:
>
> - Tensorflow: [https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03](https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03)
> - PaddlePaddle: [https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0-rc1/install/compile/linux-compile.html](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0-rc1/install/compile/linux-compile.html)
> - mmdetection: [https://mmdetection.readthedocs.io/en/latest/get_started.html#installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)
> - onnxruntime-gpu: [https://github.com/microsoft/onnxruntime/blob/master/BUILD.md](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md)

Confirm all deep learning frameworks installation via:

```bash
python -c "import tensorflow as tf; print(tf.__version__); import torch; print(torch.__version__); import paddle; print(paddle.__version__); import onnxruntime as rt; print(rt.__version__); import mmdets; print(mmdet.__version__)"
```

Then install other necessary libraries via:

```bash
pip install -r requirements.txt
```

### Enjoy!

```bash 
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to see the main page!

## Performance

Inference time consumption is highly related with following factors:

- Complexity of table structure
- Number of OCR blocks
- Resolution of selected image

A typical inference time consumption is shown in Demo Video.

## Want to contribute?

### Contribute a new cell extractor

In [`boardered/extractor.py`](https://github.com/MrZilinXiao/Hyper-Table-OCR/blob/main/boardered/extractor.py), we define a `TraditionalExtractor` based on traditional computer vision techniques and a `UNetExtractor` based on UNet pixel-level sematic segmentation model. Feel free to derive from the following abstract class:

```python
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
```

### Contribute a new OCR Module

Located in [`ocr/__init__.py`](https://github.com/MrZilinXiao/Hyper-Table-OCR/blob/main/ocr/__init__.py), you should build a custom OCR handler deriving from `OCRHandler`.

```python
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
```

### Contribute to the process pipeline

[`WebHandler.pipeline()`](https://github.com/MrZilinXiao/Hyper-Table-OCR/blob/0cda4e9c1fafadb6e375c1bfd5fc54e10d3f8c8e/web/__init__.py#L111) in `web/__init__.py`

## Future Plans
- [ ] Speed up inference via async-processing on dual GPUs.

## Authors

*This project is a participator of the 1st MegMeet Cup Technology Innovation Competition of Sichuan University. Huangwei Wu([@ndwuhuangwei](https://github.com/ndwuhuangwei)) is our team leader, while he, Zilin Xiao([@MrZilinXiao](https://github.com/MrZilinXiao)) and Ruinan Fan([@ruinanfan](https://github.com/ruinanfan)) all act as key developers of this project. It's impossible to complete this work without their effort.*

**Congratulations! This project earns a GRAND PRIZE(2 out of 72 participators) of the aforementioned competition!**

## Acknowledgement

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): Multilingual, awesome, leading, and practical OCR tools supported by Baidu.
- [ChineseOCR_lite](https://github.com/ouyanghuiyu/chineseocr_lite): Super light OCR inference tool kit.
- [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet): An automatic table recognition method for interpretation of tabular data in document images.
- [pytorch-hed](https://github.com/sniklaus/pytorch-hed): An unofficial implementation of Holistically-Nested Edge Detection using PyTorch.
