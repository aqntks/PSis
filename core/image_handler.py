import cv2
import numpy as np


class ImagePack:
    def __init__(self, path, img_size=640, stride=32):
        self.o_img = cv2.imread(path)  # 원본
        assert self.o_img is not None, '이미지를 찾을 수 없습니다 ' + path
        self.n_img = self.o_img  # 현재 이미지
        self.img_size = img_size
        self.stride = stride
        self.t_img = self.img2pyt(self.n_img)  # 검출용 이미지

    def img2pyt(self, imgO):
        img = letterbox(imgO, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img

    def reset(self, img_size, stride):
        self.img_size = img_size
        self.stride = stride
        self.t_img = self.img2pyt(self.n_img)

    def crop(self, rect, im0s):
        x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        img_crop = im0s[y1:y2, x1:x2]
        return img_crop

    def setImg(self, img):
        self.n_img = img
        self.t_img = self.img2pyt(self.n_img)

    def getImg(self):
        return self.t_img, self.n_img

    def setYCrop(self):
        self.n_img = self.crop((0, int(self.n_img.shape[0] / 2), self.n_img.shape[1], self.n_img.shape[0]),
                                       self.n_img)
        self.t_img = self.img2pyt(self.n_img)

        return self.t_img, self.n_img

    def getOImg(self):
        return self.o_img


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)