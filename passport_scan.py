import cv2
import torch
import numpy as np
from numpy import random

import argparse

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

from perspective_transform.perspective_transform import perspective


def main(opt):
    weights, images, img_size, confidence, iou, device = \
        opt.weights, opt.img, opt.img_size, opt.conf, opt.iou, opt.device

    # 디바이스 세팅
    device = select_device(device)  # 첫번째 gpu 사용
    half = device.type != 'cpu'  # gpu + cpu 섞어서 사용

    # 모델 로드
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if half:
        model.half()

    # 데이터 세팅
    dataset = LoadImages(images, img_size=img_size, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
    for path, img, im0s, vid_cap in dataset:
        # 이미지 정규화
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론 & NMS 적용
        prediction = model(img, augment=False)[0]
        prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)

        if len(prediction[0]) < 75:
            print('### Detection Fail ###')
            continue

        # 검출 값 처리
        for i, det in enumerate(prediction):
            if len(det):
                obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                mrz_rect = None

                # mrz detect
                for *rect, conf, cls in det:
                    if names[int(cls)] == 'mrz':
                        mrz_rect = rect
                        break

                if mrz_rect is None:

                    print("mrz 안나옴")
                    continue

                # mrz sort
                result, mrzStr, = '', []
                for *rect, conf, cls in det:
                    if (rect[0] > mrz_rect[0]) and (rect[1] > mrz_rect[1]) and (rect[2] < mrz_rect[2]) and (
                            rect[3] < mrz_rect[3]):
                        cls_name = names[int(cls)] if names[int(cls)] != 'sign' else '<'
                        mrzStr.append((rect[0], rect[1], cls_name))

                mrzStr.sort(key=lambda x: x[1])
                mrzFirst, mrzSecond = mrzStr[0:44], mrzStr[44:]
                mrzFirst.sort(key=lambda x: x[0])
                mrzSecond.sort(key=lambda x: x[0])

                for x, y, mrz_cls in mrzFirst:
                    result += mrz_cls
                result += '\n'
                for x, y, mrz_cls in mrzSecond:
                    result += mrz_cls

            showImg(det, names, im0s, colors)
        # print('\n' + result)

        surName, givenNames = spiltName(result[5:44])
        passportType = typeCorrection(mrzCorrection(result[0:2].replace('<', ''), 'dg2en'))
        issuingCounty = nationCorrection(mrzCorrection(result[2:5], 'dg2en'))
        nationality = nationCorrection(mrzCorrection(result[55:58], 'dg2en'))
        passportNo = result[45:54].replace('<', '')
        sur = mrzCorrection(surName.replace('<', ' ').strip(), 'dg2en')
        given = mrzCorrection(givenNames.replace('<', ' ').strip(), 'dg2en')
        personalNo = mrzCorrection(result[73:80].replace('<', ''), 'en2dg')
        birth = mrzCorrection(result[58:64].replace('<', ''), 'en2dg')
        sex = sexCorrection(mrzCorrection(result[65].replace('<', ''), 'dg2en'))
        expiry = mrzCorrection(result[66:72].replace('<', ''), 'en2dg')

        # result print
        print("\n\n--------- Passport Scan Result ---------")
        print('Type            :', passportType)
        print('Issuing county  :', issuingCounty)
        print('Passport No.    :', passportNo)
        print('Surname         :', sur)
        print('Given names     :', given)
        print('Nationality     :', nationality)
        # print('Personal No.    :', personalNo)
        print('Date of birth   :', birth)
        print('Sex             :', sex)
        print('Date of expiry  :', expiry)
        print("----------------------------------------\n")
        cv2.waitKey(0)


# 국가 보정
def nationCorrection(value):
    # 국가명 파일 로드
    f = open("weights/nationality.txt", 'r')
    nationality = []
    while True:
        line = f.readline()
        if not line: break
        nationality.append(line)
    f.close()

    # 글자수 체크
    if len(value) != 3: return value

    # 국가명 확인
    for nation in nationality:
        if nation == value:
            return value

    count, resultNation = 0, ''

    # 앞에 두자리 맞으면 비슷한 국가 출력
    strFront = value[0:2]
    for nation in nationality:
        if len(nation) != 3: continue
        if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strFront == nation[0:2]:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    count, resultNation = 0, ''

    # 뒤의 두자리 맞으면 비슷한 국가 출력
    strBack = value[1:]
    for nation in nationality:
        if len(nation) != 3: continue
        if count > 1: return nation   # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strBack == nation[1:]:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    count, resultNation = 0, ''

    # 중간만 틀렸을 때 비슷한 국가 출력
    strMiddle = value[0] + value[2]
    for nation in nationality:
        nation = nation[0] + nation[2]
        if len(nation) != 3: continue
        if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
        if strMiddle == nation:
            count += 1
            resultNation = nation

    if count == 1: return resultNation
    return value


# mrz 영어, 숫자 보정
def mrzCorrection(value, flag):
    if flag == 'en2dg':
        return value.replace('O', '0').replace('Q', '0').replace('U', '0').replace('D', '0')\
            .replace('I', '1').replace('Z', '2').replace('B', '3').replace('A', '4').replace('S', '5')
    else:
        return value.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('3', 'B')\
            .replace('4', 'A').replace('8', 'B')


# 성별 보정
def sexCorrection(value):
    return value.replace('P', 'F').replace('T', 'F').replace('N', 'M')


# 여권 타입 보정
def typeCorrection(value):
    return value.replace('FM', 'PM').replace('PN', 'PM')


# 이름 Surname, GivenName 분리
def spiltName(name):
    nameCheck, nameBool = 0, False
    surName, givenNames = '', ''
    for s in name:
        if s == '<':
            nameCheck += 1
        else:
            if nameCheck == 1: nameCheck = 0

        if nameCheck == 2 and nameBool is True:
            break
        elif nameCheck == 2:
            nameCheck = 0
            nameBool = True
        elif nameBool is False:
            surName += s
        else:
            givenNames += s

    return surName, givenNames


# 이미지 크롭
def crop(rect, im0s):
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
    img_crop = im0s[y1:y2, x1:x2]
    return img_crop


# 검출 여부 확인
def nonCheck(item, obj):
    return obj[item] if item in obj else ('0', 0)


# 이미지 출력 (openCV)
def showImg(det, names, im0s, colors):
    realImg, drawImg = im0s.copy(), im0s.copy()
    for *rect, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=1)

    appendImg = np.append(realImg, drawImg, axis=1)
    cv2.imshow("result", cv2.resize(appendImg, (1616, 504)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/passport_x_v2_0617.pt')
    parser.add_argument('--img', type=str, default='data/images')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.50)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', type=str, default='cpu')
    option = parser.parse_args()
    main(opt=option)