# import cv2
# import torch
# import numpy as np
# import math
# from numpy import random
#
# import argparse
#
# from models.experimental import attempt_load
# from core.datasets import LoadImages
# from core.general import check_img_size, non_max_suppression, scale_coords
# from core.plots import plot_one_box
# from core.torch_utils import select_device, time_synchronized
#
#
# def main(opt):
#     weights, images, img_size, confidence, iou, device = \
#         opt.weights, opt.img, opt.img_size, opt.conf, opt.iou, opt.device
#
#     # 디바이스 세팅
#     device = select_device(device)  # 첫번째 gpu 사용
#     half = device.type != 'cpu'  # gpu + cpu 섞어서 사용
#
#     # 모델 로드
#     model = attempt_load(weights, map_location=device)
#     stride = int(model.stride.max())
#     img_size = check_img_size(img_size, s=stride)
#     if half:
#         model.half()
#
#     # 데이터 세팅
#     dataset = LoadImages(images, img_size=img_size, stride=stride)
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#
#     # 추론 실행
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
#     for path, img, im0s, vid_cap, real in dataset:
#
#         # 이미지 정규화
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # 추론 & NMS 적용
#         prediction = model(img, augment=False)[0]
#         prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)
#
#         rect_list, det_tmp = [], []
#         for _, det in enumerate(prediction):
#             obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
#             detect = det
#
#         # 검출 제대로 안된 경우 로테이션
#         if len(detect) < 50:
#             for deg in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
#                 r_im0s = cv2.rotate(real, deg)
#                 r_img = letterbox(r_im0s, img_size, stride=stride)[0]
#
#                 r_img = r_img[:, :, ::-1].transpose(2, 0, 1)
#                 r_img = np.ascontiguousarray(r_img)
#
#                 # 이미지 정규화
#                 r_img = torch.from_numpy(r_img).to(device)
#                 r_img = r_img.half() if half else r_img.float()
#                 r_img /= 255.0
#                 if r_img.ndimension() == 3:
#                     r_img = r_img.unsqueeze(0)
#
#                 # 추론 & NMS 적용
#                 r_prediction = model(r_img, augment=False)[0]
#                 r_prediction = non_max_suppression(r_prediction, confidence, iou, classes=None, agnostic=False)
#
#                 for _, det in enumerate(r_prediction):
#                     obj, det[:, :4] = {}, scale_coords(r_img.shape[2:], det[:, :4], r_im0s.shape).round()
#                     r_det = det
#                 if len(r_det) > 70:
#                     im0s = r_im0s
#                     img = r_img
#                     prediction = r_prediction
#                     detect = r_det
#                     break
#
#         for *rect, conf, cls in detect:
#             if names[int(cls)] != 'mrz':
#                 det_tmp.append((rect, conf))
#
#         rect_list = unsorted_remove_intersect_box(det_tmp)
#
#         if len(rect_list) < 2:
#             continue
#
#         # 기울기 조정
#         rect_list.sort(key=lambda x: x[0])
#         firstChar = rect_list[0] if rect_list[0][1] < rect_list[1][1] else rect_list[1]
#         lastChar = rect_list[len(rect_list) - 1] \
#             if rect_list[len(rect_list) - 1][1] < rect_list[len(rect_list) - 2][1] else rect_list[len(rect_list) - 2]
#
#         p1_x, p1_y = firstChar[0], firstChar[1]
#         p2_x, p2_y = lastChar[0], lastChar[1]
#         degree = degree_detection(p1_x, p1_y, p2_x, p2_y)
#         im0s = affine_rotation(im0s, degree)
#
#         img = letterbox(im0s, img_size, stride=stride)[0]
#
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#
#         # 이미지 정규화
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         # 추론 & NMS 적용
#         prediction = model(img, augment=False)[0]
#         prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)
#
#         # 검출 값 처리
#         for i, det in enumerate(prediction):
#             if len(det):
#                 obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
#
#                 mrz_rect = None
#
#                 # mrz detect
#                 for *rect, conf, cls in det:
#                     if names[int(cls)] == 'mrz':
#                         mrz_rect = rect
#                         break
#
#                 if mrz_rect is None:
#                     print("mrz 안나옴")
#                     continue
#
#                 # mrz sort
#                 result, mrzStr, = '', []
#                 for *rect, conf, cls in det:
#                     if (rect[0] > mrz_rect[0]) and (rect[1] > mrz_rect[1]) and (rect[2] < mrz_rect[2]) and (
#                             rect[3] < mrz_rect[3]):
#                         cls_name = names[int(cls)] if names[int(cls)] != 'sign' else '<'
#                         mrzStr.append((rect, cls_name, conf))
#
#                 mrzStr.sort(key=lambda x: x[0][1])
#
#                 # 라인단위 정렬 v2
#                 # mrzFirst, mrzSecond = sort_v2(mrzStr)
#
#                 # 라인 단위 정렬
#                 mrzFirst, mrzSecond = line_by_line_sort(mrzStr)
#
#                 # 한번에 정렬
#                 # mrzFirst, mrzSecond = all_sort(mrzStr)
#
#                 mrzFirst, mrzSecond = remove_intersect_box(mrzFirst), remove_intersect_box(mrzSecond)
#
#                 firstLine, secondLine = "", ""
#                 for rect, mrz_cls, conf in mrzFirst:
#                     firstLine += mrz_cls
#                 for rect, mrz_cls, conf in mrzSecond:
#                     secondLine += mrz_cls
#
#                 if len(firstLine) < 44:
#                     for i in range(len(firstLine), 44):
#                         firstLine += '<'
#
#                 if len(secondLine) < 44:
#                     for i in range(len(secondLine), 44):
#                         secondLine += '<'
#
#             showImg(det, names, im0s, colors, real)
#
#         surName, givenNames = spiltName(firstLine[5:44])
#         passportType = typeCorrection(mrzCorrection(firstLine[0:2].replace('<', ''), 'dg2en'))
#         issuingCounty = nationCorrection(mrzCorrection(firstLine[2:5], 'dg2en'))
#         sur = mrzCorrection(surName.replace('<', ' ').strip(), 'dg2en')
#         given = mrzCorrection(givenNames.replace('<', ' ').strip(), 'dg2en')
#
#         passportNo = secondLine[0:9].replace('<', '')
#         nationality = nationCorrection(mrzCorrection(secondLine[10:13], 'dg2en'))
#         birth = mrzCorrection(secondLine[13:19].replace('<', ''), 'en2dg')
#         sex = sexCorrection(mrzCorrection(secondLine[20].replace('<', ''), 'dg2en'))
#         expiry = mrzCorrection(secondLine[21:27].replace('<', ''), 'en2dg')
#         personalNo = mrzCorrection(secondLine[28:35].replace('<', ''), 'en2dg')
#
#         # result print
#         print("\n\n--------- Passport Scan Result ---------")
#         print('Type            :', passportType)
#         print('Issuing county  :', issuingCounty)
#         print('Passport No.    :', passportNo)
#         print('Surname         :', sur)
#         print('Given names     :', given)
#         print('Nationality     :', nationality)
#         # print('Personal No.    :', personalNo)
#         print('Date of birth   :', birth)
#         print('Sex             :', sex)
#         print('Date of expiry  :', expiry)
#         print("----------------------------------------\n")
#         cv2.waitKey(0)
#
#
# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)
#
#
# # 검출 박스 상자의 겹친 비율
# def compute_intersect_ratio(rect1, rect2):
#     x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
#     x3, y3, x4, y4 = rect2[0], rect2[1], rect2[2], rect2[3]
#
#     if x2 < x3: return 0
#     if x1 > x4: return 0
#     if y2 < y3: return 0
#     if y1 > y4: return 0
#
#     left_up_x = max(x1, x3)
#     left_up_y = max(y1, y3)
#     right_down_x = min(x2, x4)
#     right_down_y = min(y2, y4)
#
#     width = right_down_x - left_up_x
#     height = right_down_y - left_up_y
#
#     original = (y2 - y1) * (x2 - x1)
#     intersect = width * height
#
#     ratio = int(intersect / original * 100)
#
#     return ratio
#
#
# # 겹친 상자 제거 (30% 이상)
# def remove_intersect_box(mrzLine):
#     i, line = 0, mrzLine.copy()
#     while True:
#         if i > len(line) - 2: break
#         if compute_intersect_ratio(line[i][0], line[i+1][0]) > 30:
#             lose = i if line[i][2] < line[i+1][2] else i+1
#             del line[lose]
#         else: i += 1
#
#     return line
#
#
# # 겹친 상자 제거 (30% 이상) - 정렬 하기 힘든 경우
# def unsorted_remove_intersect_box(lists):
#     for i in range(0, len(lists)-1):
#         if i > len(lists)-2: break
#         for y in range(i+1, len(lists)-1):
#             if y > len(lists)-1: break
#             if compute_intersect_ratio(lists[i][0], lists[y][0]) > 30:
#                 if lists[i][1] > lists[y][1]:
#                     del lists[y]
#                     y -= 1
#                 else:
#                     del list[i]
#                     i -= 1
#
#     result = []
#     for l in lists:
#         result.append(l[0])
#     return result
#
#
# # 어파인 로테이션
# def affine_rotation(src, angle):
#     cp = (src.shape[1] / 2, src.shape[0] / 2)
#     affine_mat = cv2.getRotationMatrix2D(cp, angle, 1)
#
#     dst = cv2.warpAffine(src, affine_mat, (0, 0))
#     return dst
#
#
# # 각도 검출
# def degree_detection(p1_x, p1_y, p2_x, p2_y):
#     radian = math.atan2(p2_y - p1_y, p2_x - p1_x)
#     return radian * 180 / np.pi
#
#
# # 라인단위 정렬
# def sort_v2(mrzStr):
#     mrzStr.sort(key=lambda x: x[0][0])
#
#     firstLine_firstChar = mrzStr[0] if mrzStr[0][0][1] < mrzStr[1][0][1] else mrzStr[1]
#     firstLine_lastChar = mrzStr[len(mrzStr) - 1] if mrzStr[len(mrzStr) - 1][0][1] < mrzStr[len(mrzStr) - 2][0][1] else mrzStr[len(mrzStr) - 2]
#     standard = firstLine_firstChar if firstLine_firstChar[0][1] > firstLine_lastChar[0][1] else firstLine_lastChar
#
#     mrzFirst, mrzSecond = [], []
#     for c in mrzStr:
#         if c[0][1] <= standard[0][1]:
#             mrzFirst.append(c)
#         else:
#             mrzSecond.append(c)
#
#     mrzFirst.sort(key=lambda x: x[0][0])
#     mrzSecond.sort(key=lambda x: x[0][0])
#
#     return mrzFirst, mrzSecond
#
#
# # 라인단위 정렬
# def line_by_line_sort(mrzStr):
#     middleChar, mrzFirst, mrzSecond = mrzStr[0], [], []
#     for c in mrzStr:
#         if c[0][1] < middleChar[0][3]:
#             mrzFirst.append(c)
#         else:
#             mrzSecond.append(c)
#
#     mrzFirst.sort(key=lambda x: x[0][0])
#     mrzSecond.sort(key=lambda x: x[0][0])
#
#     return mrzFirst, mrzSecond
#
#
# # 검출 값 한꺼번에 정렬
# def all_sort(mrzStr):
#     mrzFirst, mrzSecond = mrzStr[0:44], mrzStr[44:]
#     mrzFirst.sort(key=lambda x: x[0])
#     mrzSecond.sort(key=lambda x: x[0])
#
#     return mrzFirst, mrzSecond
#
#
# # 국가 보정
# def nationCorrection(value):
#     # 국가명 파일 로드
#     f = open("weights/nationality.txt", 'r')
#     nationality = []
#     while True:
#         line = f.readline()
#         if not line: break
#         nationality.append(line)
#     f.close()
#
#     # 글자수 체크
#     if len(value) != 3: return value
#
#     # 국가명 확인
#     for nation in nationality:
#         if nation == value:
#             return value
#
#     strFront = value[0:2]
#     strBack = value[1:]
#     strMiddle = value[0] + value[2]
#     if strFront == 'KO': return 'KOR'
#     if strBack == 'OR': return 'KOR'
#     if strMiddle == 'KR': return 'KOR'
#
#     count, resultNation = 0, ''
#
#     # 앞에 두자리 맞으면 비슷한 국가 출력
#     for nation in nationality:
#         if len(nation) != 3: continue
#         if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
#         if strFront == nation[0:2]:
#             count += 1
#             resultNation = nation
#
#     if count == 1: return resultNation
#     count, resultNation = 0, ''
#
#     # 뒤의 두자리 맞으면 비슷한 국가 출력
#     for nation in nationality:
#         if len(nation) != 3: continue
#         if count > 1: return nation   # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
#         if strBack == nation[1:]:
#             count += 1
#             resultNation = nation
#
#     if count == 1: return resultNation
#     count, resultNation = 0, ''
#
#     # 중간만 틀렸을 때 비슷한 국가 출력
#     for nation in nationality:
#         nation = nation[0] + nation[2]
#         if len(nation) != 3: continue
#         if count > 1: return nation  # 오탐일 경우 재 탐색하는 기능 여부에 따라 수정
#         if strMiddle == nation:
#             count += 1
#             resultNation = nation
#
#     if count == 1: return resultNation
#     return value
#
#
# # mrz 영어, 숫자 보정
# def mrzCorrection(value, flag):
#     if flag == 'en2dg':
#         return value.replace('O', '0').replace('Q', '0').replace('U', '0').replace('D', '0')\
#             .replace('I', '1').replace('Z', '2').replace('B', '3').replace('A', '4').replace('S', '5')
#     else:
#         return value.replace('0', 'O').replace('1', 'I').replace('2', 'Z').replace('3', 'B')\
#             .replace('4', 'A').replace('8', 'B')
#
#
# # 성별 보정
# def sexCorrection(value):
#     return value.replace('P', 'F').replace('T', 'F').replace('N', 'M')
#
#
# # 여권 타입 보정
# def typeCorrection(value):
#     return value.replace('FM', 'PM').replace('PN', 'PM')
#
#
# # 이름 Surname, GivenName 분리
# def spiltName(name):
#     nameCheck, nameBool = 0, False
#     surName, givenNames = '', ''
#     for s in name:
#         if s == '<':
#             nameCheck += 1
#         else:
#             if nameCheck == 1: nameCheck = 0
#
#         if nameCheck == 2 and nameBool is True:
#             break
#         elif nameCheck == 2:
#             nameCheck = 0
#             nameBool = True
#         elif nameBool is False:
#             surName += s
#         else:
#             givenNames += s
#
#     return surName, givenNames
#
#
# # 이미지 크롭
# def crop(rect, im0s):
#     x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
#     img_crop = im0s[y1:y2, x1:x2]
#     return img_crop
#
#
# # 검출 여부 확인
# def nonCheck(item, obj):
#     return obj[item] if item in obj else ('0', 0)
#
#
# # 이미지 출력 (openCV)
# def showImg(det, names, im0s, colors, real):
#     realImg, drawImg = real.copy(), im0s.copy()
#     for *rect, conf, cls in reversed(det):
#         label = f'{names[int(cls)]} {conf:.2f}'
#         plot_one_box(rect, drawImg, label=label, color=colors[int(cls)], line_thickness=1)
#
#     appendImg = np.append(cv2.resize(realImg, (drawImg.shape[1], drawImg.shape[0])), drawImg, axis=1)
#     cv2.imshow("result", cv2.resize(appendImg, (1280, 400)))
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='weights/passport_m.pt')
#     parser.add_argument('--img', type=str, default='data/test_img')
#     parser.add_argument('--img-size', type=int, default=640)
#     parser.add_argument('--conf', type=float, default=0.25)
#     parser.add_argument('--iou', type=float, default=0.45)
#     parser.add_argument('--device', type=str, default='cpu')
#     option = parser.parse_args()
#     main(opt=option)