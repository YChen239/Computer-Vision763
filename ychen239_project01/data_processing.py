import numpy as np
import os
from PIL import Image



def getRes_det(fdet):
    detFile = open(fdet, 'r')
    det = detFile.readlines()
    detFile.close()
    res = []
    for idx in range(len(det)):
        if '/' in det[idx]:
            im_res = {}
            im_res['name'] = det[idx].strip()
            num = int(det[idx + 1])
            im_res['num'] = num
            coord = []
            for i in range(idx + 2, idx + 2 + num):
                coord_str = det[i].split()
                coord_float = [float(i) for i in coord_str]
                coord.append(coord_float)
            im_res['coord'] = np.array(coord)
            res.append(im_res)
    return res




def showRes(res,i):
    import random
    dets = res['coord']
    im_name = os.path.join('C:/Study/19Spring/ECE763/hw&pro/project1', 'originalPics', res['name'] + '.jpg')
    im = Image.open(im_name)
    (width,height) = im.size

    x = dets[0, 3]
    y = dets[0, 4]
    w = dets[0, 0]
    h = dets[0, 1]
    region = im.crop((x - h, y - w, x + h, y + w))
    region = region.resize((60, 60))

    y_0 = random.randint(0, height - 60)
    x_0 = random.randint(0, width - 60)

    # # y_0 = random.randint(0, height - 60)
    # # x_0 = random.randint(0, width - 60)
    #
    while( x-h <= x_0 < x-h+50 and y-w <= y_0 < y-w+50):
        y_0 = random.randint(0, height-60)
        x_0 = random.randint(0, width-60)

    random_img = im.crop((x_0, y_0, x_0 + 60, y_0 + 60))


    if i<1000:
        region.save("./data/training/face/face{:0>2}.png".format(i + 1))
        random_img.save("./data/training/non-face/non-face{:0>2}.png".format(i + 1))
    else:
        region.save("./data/test/face/face{:0>2}.png".format(i + 1-1000))
        random_img.save("./data/test/non-face/non-face{:0>2}.png".format(i + 1 -1000))




ann_name = []  # annotation file name
for i in range(10):
    ann_name.append('C:/Study/19Spring/ECE763/hw&pro/project1/FDDB-folds/FDDB-fold-{:0>2}-ellipseList.txt'.format(i + 1))


ann = []
for i in range(10):
    ann_det = getRes_det(ann_name[i])
    ann.extend(ann_det)
for i in range(1100):
    temp_ann = ann[i]
    showRes(temp_ann,i)