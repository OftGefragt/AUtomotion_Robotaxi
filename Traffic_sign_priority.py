
'''
detections [x, y, w, h, conf, class_id] çıktısı alınıyor 

import pyzed.sl as sl
import cv2
import torch
import numpy as np
from collections import deque
'''


class prioritytrfc:

    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh
        self.detections = [] 


    def add_detection(self, detection):
        x, y, w, h, conf, class_id = detection
        if conf < self.conf_thresh:
            return 
    
        self.detections.append(detection)

        #descending order bu
        self.detections.sort(key=lambda d: d[2]*d[3], reverse=True)

    def get_top_priority(self):

        if not self.detections:
            return None
        return self.detections[0] #pipeline'da varsa en büyük boyutluyu döndür. (ALAN TEMELLİ)

    def clear(self):
        self.detections.clear()
''' 
inference yapılacak model yüklenir.

from ultralytics import YOLO

model = YOLO('/.pt adresi ')

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img

def run_yolo(frame):
    img = preprocess(frame)
    results = model(img)  # inference run'la
    detections = []
    for result in results:
      boxes = result.boxes
      for box in boxes:
        x, y, w, h = box.xywh[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        detections.append([x, y, w, h, conf, cls])
    return detections 



def main():
  
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60

#real life testlerine göre 60 fps iyi, kamera 720p izin veriyor ,yolo 640 resize ettiği için en yakın ölçüyü seçiyoruz.

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open error: {err}")
        return

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    queue = prioritytrfc(conf_thresh=0.5)

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()  #soldan img çekiyoruz rgb cam mantığı ile, yolo stereo almayacak.

                detections = run_yolo(frame)
                queue.clear()
                for det in detections:
                    queue.add_detection(det)

                top_sign = queue.get_top_priority()
                if top_sign:
                    x, y, w, h, conf, cls = top_sign
                
  
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f'Cls {cls} {conf:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.imshow("ZED Left View with Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


'''




