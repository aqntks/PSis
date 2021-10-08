import os
import yaml
import shutil
import datetime
import argparse

from models.experimental import attempt_load
from core.passport_scan import *

import requests
import json


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def jsonSend(passport, path, err_code):
    passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry, personalNo = passport.all()

    data = {}
    ppscan_result = {}
    fileName = path.split("\\")[-1]
    dir = path[:path.rfind("\\")]
    if len(fileName.split('_')) > 3:
        merchant_id = fileName.split('_')[0]
        person_order = fileName.split('_')[1]
        adt_chd = fileName.split('_')[2]
    else:
        merchant_id = ""
        person_order = ""
        adt_chd = ""
        err_code = 55
    url = "https://www.oye.co.kr/CodeOne/ReturnFile.html"

    if err_code == 99:
        data['err_code'] = err_code
        data['merchant_id'] = merchant_id
        data['person_order'] = person_order
        data['adt_chd'] = adt_chd
        ppscan_result['passport_no'] = ""
        ppscan_result['sur_name'] = ""
        ppscan_result['given_name'] = ""
        ppscan_result['nationality'] = ""
        ppscan_result['sex'] = ""
        ppscan_result['birth_date'] = ""
        ppscan_result['expiry_date'] = ""
        data['ppscan_result'] = ppscan_result

        jsondata = json.dumps(data, indent="\t")
        response = requests.post(url, json=jsondata)
        st_code = response.status_code
    else:

        data = {}
        ppscan_result = {}

        data['err_code'] = err_code
        data['merchant_id'] = merchant_id
        data['person_order'] = person_order
        data['adt_chd'] = adt_chd
        ppscan_result['passport_no'] = passportNo
        ppscan_result['sur_name'] = sur
        ppscan_result['given_name'] = given
        ppscan_result['nationality'] = nationality
        ppscan_result['sex'] = sex
        ppscan_result['birth_date'] = birth
        ppscan_result['expiry_date'] = expiry
        data['ppscan_result'] = ppscan_result

        jsondata = json.dumps(data, indent="\t")
        response = requests.post(url, json=jsondata)
        st_code = response.status_code
        print(response.text)
        print(st_code)


def main(arg):
    gpu, mode = arg.gpu, arg.mode
    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    weights, images, img_size, confidence, iou = \
        config['weights'], config['images'], config['img_size'], config['confidence'], config['iou']
    option = (img_size, confidence, iou)

    # 디바이스 세팅
    if gpu == -1:
        dev = 'cpu'
    else:
        dev = f'cuda:{gpu}'
    device = torch.device(dev)

    # 모델 세팅
    model = attempt_load(weights, map_location=device)

    # 반복문 시작
    oldHour = 30
    while True:
        filelist = os.listdir(images)
        if filelist:
            for f in filelist:
                ext = f.split('.')[-1].lower()
                image = images + "/" + f
                if ext in img_formats:
                    path = f'images/{f}'
                    err_code = 10
                    try:
                        passport = detect(image, model, device, option, mode=mode)
                    except Exception as e:
                        print(e)
                        jsonSend(passport, path, err_code=40)
                        shutil.move(image, f'D:/temp/passport_mv/{f}')

                    if passport is None:
                        passport = Passport('', '', '', '', '', '', '', '', '', '')
                        err_code = 99

                    jsonSend(passport, path, err_code=err_code)
                    shutil.move(image, f'D:/temp/passport_mv/{f}')

                else:
                    print(f"이미지가 아닌 파일 : {f}")
                    shutil.move(image, f'D:/temp/passport_mv/{f}')
            filelist.clear()
        else:
            time.sleep(1)
            nowH = datetime.datetime.now()
            if nowH.minute == 0:
                if nowH.hour != oldHour:
                    oldHour = nowH.hour
                    print(f"{nowH} : 프로그램 작동중...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='print')
    opt = parser.parse_args()
    main(arg=opt)
