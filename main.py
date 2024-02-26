from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from multiprocessing import set_start_method
from multiprocessing import Process
import cv2
import numpy as np
from matplotlib import pyplot as plt

# TRAINING for one of the models
def train(model_yolo, model_name):
    # Load the model
    model = YOLO(f'models/{model_yolo}')
    
    # Training.
    results = model.train(
        data=f'datasets/{model_name}/data.yaml',
        imgsz=640,
        epochs=10,
        batch=8,
        workers=8,
        name=model_name
    )

    return results

# PREDICTIONS based on a given test image
def predict(model_name, save_name, img, conf):
    model = YOLO(f'runs/detect/{model_name}/weights/best.pt')
    
    results = model.predict(
        source=img,
        conf=conf, 
        show_labels=False
    )

    # Process results list
    for i, result in enumerate(results):
        result.save(filename=f'results/{model_name}/{save_name[i]}_{conf}.jpg')  # save to disk

    return img

## Testing some borderline preprocessing
## Tried to see what else was possible outside of Neural Nets
def preprocessing(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    print('GaussianBlur')
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # sobel = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobel = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    # sobel = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    # Detect edges using the Canny edge detector
    print('Canny')
    edges = cv2.Canny(blurred, 30, 100)

    # Apply binary thresholding to segment the image
    print('threshold')
    _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    print('bitwise_not')
    binary = cv2.bitwise_not(binary)

    # Find contours
    print('findContours')
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    print('drawContours')
    segmented_image = img.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 3)

    # Display the original image and the segmented image
    # comp = cv2.addWeighted(img, 1, edges, 1, 0.0)
    # cv2.imshow('.', comp)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predictCombined():
    img_s = cv2.imread('datasets/screws.jpg')
    img_b = cv2.imread('datasets/battery.png')

    model_screws = 'runs/detect/screws/weights/best.pt'
    model_batteries = 'runs/detect/batteries/weights/best.pt'

    # results = model(img)
    model_screws = YOLO(model_screws)
    model_batteries = YOLO(model_batteries)

    screws1 = model_screws.predict(source=img_s, conf=0.01, show_labels=False)
    screws2 = model_batteries.predict(source=img_s, conf=0.25, show_labels=False)
    batteries1 = model_screws.predict(source=img_b, conf=0.01, show_labels=False)
    batteries2 = model_batteries.predict(source=img_b, conf=0.25, show_labels=False)

    annotator = Annotator(img_s)
    
    for r in screws1:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            annotator.box_label(b, model_screws.names[1], color=(36, 62, 54))
    for r in screws2:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model_batteries.names[int(c)], color=(124, 169, 130))      
    img_screws = annotator.result()  

    annotator = Annotator(img_b)

    for r in batteries1:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            annotator.box_label(b, model_screws.names[1], color=(36, 62, 54))
    for r in batteries2:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model_batteries.names[int(c)], color=(124, 169, 130))

    img_batteries = annotator.result()  
    cv2.imwrite('results/combined/screws.jpg', img_screws)
    cv2.imwrite('results/combined/batteries.jpg', img_batteries)

if __name__ == '__main__':
    # ## Uncomment one of the following
    model_name = 'screws'
    # model_name = 'batteries'

    # ## Training and Cross Validation
    # ## For Screws and Batteries models
    train('yolov8n.pt', model_name)

    # Test Images
    img = ['datasets/screws.jpg', 'datasets/battery.png']

    # ## Test Batteries Detection Model
    # predict(model_name, ['screws','batteries'], img, conf=0.01)
    # predict(model_name, ['screws','batteries'], img, conf=0.05)
    # predict(model_name, ['screws','batteries'], img, conf=0.10)
    # predict(model_name, ['screws','batteries'], img, conf=0.25)

    # ## Test Screws Detection Model
    # model_name = 'screws'
    # predict(model_name, ['screws','batteries'], img, conf=0.01)
    # predict(model_name, ['screws','batteries'], img, conf=0.05)
    # predict(model_name, ['screws','batteries'], img, conf=0.10)
    # predict(model_name, ['screws','batteries'], img, conf=0.25)

    # ## Makes the predictions based on both of the models and plots the image with BBs from both results
    # predictCombined()

    ## Testing some borderline preprocessing
    ## Tried to see what else was possible outside of Neural Nets
    # predictCombined(img)