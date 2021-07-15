import os
from core.model_set import attempt_load
from core.passport_scan import *

# 설정
weights = 'weights/passport_m.pt'
images = 'data/test_img'
img_size = 640
confidence = 0.25
iou = 0.45
opt = (img_size, confidence, iou)

# 디바이스 세팅
device = torch.device('cpu')
half = device.type != 'cpu'

model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
s_img_size = check_img_size(img_size, s=stride)
if half:
    model.half()


list_dir = os.listdir(images)

for p in list_dir:
    path = images + '/' + p
    print(p)
    passport = detect(path, model, device, opt)

    if passport:
        passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry, personalNo = passport.all()

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
    else:
        print('실패')
    cv2.waitKey(0)



