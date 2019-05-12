import numpy as np
import cv2


def dataExtract(i, data_dir):
    data = []
    for i in range(i):
        im = cv2.imread(data_dir.format(i + 1)+ '.png')

        b, g, r = cv2.split(im)
        b = b.reshape(1,-1)
        g = g.reshape(1,-1)
        r = r.reshape(1,-1)

        x = np.hstack((b,g))
        x = np.hstack((x,r))
        x = np.append(x,1)
        data.append(im)

    data = np.array(data)
    return(data)



training_nonface_data = []
for i in range(1000):
    im = cv2.imread('./data/training/non-face/non-face{:0>2}'.format(i + 1) + '.png')

    b, g, r = cv2.split(im)
    b = b.reshape(1, -1)
    g = g.reshape(1, -1)
    r = r.reshape(1, -1)

    x = np.hstack((b, g))
    x = np.hstack((x, r))
    x = np.append(x, 0)
    training_nonface_data.append(im)


training_nonface_data = np.array(training_nonface_data)




test_face_data = []
for i in range(100):
    im = cv2.imread('./data/test/face/face{:0>2}'.format(i + 1)+ '.png')

    b, g, r = cv2.split(im)
    b = b.reshape(1,-1)
    g = g.reshape(1,-1)
    r = r.reshape(1,-1)

    x = np.hstack((b,g))
    x = np.hstack((x,r))
    x = np.append(x,1)
    test_face_data.append(x)


test_nonface_data = []
for i in range(100):
    im = cv2.imread('./data/test/non-face/non-face{:0>2}'.format(i + 1) + '.png')

    b, g, r = cv2.split(im)
    b = b.reshape(1, -1)
    g = g.reshape(1, -1)
    r = r.reshape(1, -1)

    x = np.hstack((b, g))
    x = np.hstack((x, r))
    x = np.append(x, 0)
    test_nonface_data.append(x)

test_face_data = np.array(test_face_data)
test_nonface_data = np.array(test_nonface_data)
test_data = np.vstack((test_face_data,test_nonface_data))

