from ultralytics import YOLO
from multiprocessing import set_start_method
from multiprocessing import Process
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def train(path):
    # Load the model
    model = YOLO(path)
    
    # Training.
    results = model.train(
        data='datasets/batteries/data.yaml',
        imgsz=640,
        epochs=10,
        batch=8,
        # device=0,
        workers=8,
        name='batteries'
    )

def predict(path, img, conf):
    model = YOLO(path)
    # results = model(img)
    results = model.predict(
        # task='detect',
        source=img,
        # save=True,
        conf=conf, 
        show_labels=False
    )

    # Process results list
    for i, result in enumerate(results):
        # result.save(filename=f'res/screws_{conf}.jpg')  # save to disk
        result.save(filename=f'test/test{i}.jpg')  # save to disk

def preprocessing(path):
    img = cv.imread(path) 
    # img = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    # _,th1 = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #             cv.THRESH_BINARY,17,2)
    # th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv.THRESH_BINARY,17,2)
    # images = [img, th1, th2, th3]
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #         'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    
    # for i in range(4):
    #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()

    # edges = cv.Canny(img,64,128)
    # img = cv.bitwise_and(img, img, mask=edges)
    # img = cv.addWeighted(raw, 0.5, cv.cvtColor(edges, cv.COLOR_GRAY2RGB), 0.5, 0.0)
    # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img

if __name__ == '__main__':
    img = ['datasets/screws.jpg', 'datasets/battery.png']

    # train('yolov8n.pt')
    # preprocessing(img)
    predict('runs/detect/batteries/weights/best.pt', img, conf=0.05)