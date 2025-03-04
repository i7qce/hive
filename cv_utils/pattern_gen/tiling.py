import numpy as np
import cv2

def place_square(img, center, sidelength):
    cv2.rectangle(img, (center[0]-sidelength, center[1]-sidelength), (center[0]+sidelength, center[1]+sidelength), (255,255,255),-1)

def place_circle(img, center, radius):
    cv2.circle(img, center, radius, (255,255,255),-1)

def generate_random_data():
    img = np.zeros((512,512,3), np.uint8)
    label = np.zeros((512,512,3), np.uint8)

    for _ in range(50):
        x, y = np.random.randint(24,488,size=(2))
        place_square(img, (y,x), 10)
    
    # generate defect + label
    for _ in range(10):
        x, y = np.random.randint(24,488,size=(2))
        place_circle(img, (y,x), 10)
        place_circle(label, (y,x), 10)


    return img, (label[:,:,0]/255).astype(np.uint8)

def example_placer(img, label, loc):
    if np.random.randint(0,100) > 95:
        place_circle(img, loc, 10)
        place_circle(label, loc, 10)
    else:
        place_square(img, loc, 10)

def generate_periodic_data(input_size = 512, period = 30, placer=None):
    
    img = np.zeros((input_size,input_size,3), np.uint8)
    label = np.zeros((input_size,input_size,3), np.uint8)

    
    XX, YY = np.meshgrid(np.arange(0, input_size, period), np.arange(0, input_size, period))

    for y_idx in range(YY.shape[0]):
        for x_idx in range(XX.shape[1]):
            placer(img, label, (int(YY[y_idx, x_idx]), int(XX[y_idx, x_idx])))
    
    return img, (label[:,:,0]/255).astype(np.uint8)

def gen_batches(gen_fn, n):
    data_temp, labels_temp = [], []
    for _ in range(n):
        data, label = gen_fn()
        data_temp.append(data)
        labels_temp.append(label)
    data = np.stack(data_temp, axis=0)
    labels = np.stack(labels_temp, axis=0)

    return data, labels