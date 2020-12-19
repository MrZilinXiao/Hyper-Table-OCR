"""
Code for HED Edge Detection
Model definition coming from: https://github.com/sniklaus/pytorch-hed/blob/master/run.py
"""
import torch
import torch.nn as nn
import numpy as np


class HEDNet(nn.Module):
    def __init__(self, ckpt_path="./models/network-bsds500.pytorch"):
        super(HEDNet, self).__init__()
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
        self.load_weight(ckpt_path)

    def load_weight(self, filename='network-bsds500.pytorch'):
        checkpoint = torch.load(filename, map_location='cpu')
        self.load_state_dict({k.replace('module', 'net'): v for k, v in checkpoint.items()})

    def forward(self, tenInput):
        tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tenInput = torch.cat([tenBlue, tenGreen, tenRed], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))


def predict_edge(model, img: np.ndarray, device='cuda:0') -> np.ndarray:
    _input = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255)))
    w, h = _input.shape[2], _input.shape[1]
    model.eval()
    with torch.no_grad():
        output: torch.FloatTensor = model.forward(_input.to(device).view(1, 3, h, w))[0, :, :, :].cpu()
    return (output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.int8)


if __name__ == '__main__':
    # def estimate(model, tenInput):
    #
    #     intWidth = tenInput.shape[2]
    #     intHeight = tenInput.shape[1]
    #     return model(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
    # from PIL import Image

    # arguments_strIn = './test_720p.JPG'
    # arguments_strOut = './test_720p_hed.jpg'
    # tenInput = torch.FloatTensor(np.ascontiguousarray(
    #     np.array(Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (
    #                 1.0 / 255.0)))  # CHW BGR
    #
    # tenOutput = estimate(model, tenInput)  # (CHW), C = 1
    # PIL.Image.fromarray(
    #     (tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
    #     arguments_strOut)

    model = HEDNet()
    model.load_weight()

    import cv2

    ori_img = cv2.imread('./test_720p.JPG')

    edged_img = predict_edge(model, ori_img, device='cpu').astype(np.uint8)  # HW
    cv2.imwrite('./test_720p_hed_b.jpg', edged_img)
    pass
