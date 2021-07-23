import os
import yaml
from models.experimental import attempt_load
from core.passport_scan import *


# config 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
weights, images, img_size, confidence, iou, gpu = \
    config['weights'], config['images'], config['img_size'], config['confidence'], config['iou'], config['gpu']
option = (img_size, confidence, iou)

# 디바이스 세팅
if gpu == -1: dev = 'cpu'
else: dev = f'cuda:{gpu}'
device = torch.device(dev)

# 모델 세팅
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
s_img_size = check_img_size(img_size, s=stride)

# 이미지 path
list_dir = os.listdir(images)

for p in list_dir:
    path = images + '/' + p
    passport = detect(path, model, device, option, mode='show')

    if passport is None:
        passport = Passport('', '', '', '', '', '', '', '', '', '')

    passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry, personalNo = passport.all()




