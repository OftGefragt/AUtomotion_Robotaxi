'''
KARAR MEKANİZMASI SUNULMUŞTUR. BREAK VE FULLSTOP FONKSİYONLARI HARİCEN EKLENMELİDİR.

{
  "predictions": [
    {
      "x": 548,
      "y": 114.5,
      "width": 64,
      "height": 131,
      "confidence": 0.922,
      "class": "traffic_cone",
      "class_id": 0,
      "detection_id": "2e61e097-bb34-4a3f-8e10-7823fe81ac14"
    },
    {
      "x": 397.5,
      "y": 181.5,
      "width": 91,
      "height": 155,
      "confidence": 0.892,
      "class": "traffic_cone",
      "class_id": 0,
      "detection_id": "9ea2126d-a149-45dd-be1e-9602935b79f8"
    }
  ]
}
şeklinde cone detection algoritmasından veriler alınıyor olduğu senaryo için , 
bounding box boyutu x ,y w, h verisinde wxh olarak alan hesaplamasına tabidir.

class_id = 0 farz edilmiştir. 



import pyzed.sl as sl
import cv2
from ultralytics import YOLO
import numpy as np
'''

# cone fine-tuned model eklenir.
model = YOLO('best.pt')

def preprocess(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #resizing'i yolo kendi içinde yapacak.

def run_yolo(frame):
    img = preprocess(frame)
    results = model(img) #infer edilir. 
    cones = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            
            if cls == 0: #class id kontrol ediniz.
                x, y, w, h = box.xywh[0].tolist()
                conf = box.conf[0].item()
                cones.append({'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf})
    return cones

def get_largest_cone_bbox(cones):
    if not cones:
        return None
    
    largest = max(cones, key=lambda c: c['w'] * c['h']) #alana göre max bb seçilir.
    return largest

def get_depth_at_bbox(depth_mat, bbox):
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

    #merkez koordinatlarını veren sistemi kullanıyoruz (x,y,w,h) . Gerektiği durumda (x,y,x,y) kullanan sistemler için revize ediniz. 
    center_x = int(x)
    center_y = int(y)

    #stereo'dan hesaplanan derinlik verisi alınır.
    depth_value = depth_mat.get_value(center_x, center_y)[0]  # burada zed'den gelen uygun veri tipine göre koordinatların depth verisi alınacak 
    if depth_value == 0:  
        # neighborhood average ile fallback
        values = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = min(max(center_x + dx, 0), depth_mat.get_width() -1)
                py = min(max(center_y + dy, 0), depth_mat.get_height()-1)
                val = depth_mat.get_value(px, py)[0]
                if val > 0:
                    values.append(val)
        if values:
            depth_value = sum(values) / len(values)
        else:
            depth_value = None  

    return depth_value  #(metre birimler)

def fallback_dist(bbox_height_px, real_height_m, focal_length_py):
    if bbox_height_px == 0:
        return None
    return (focal_length_py * real_height_m) / bbox_height_px

def main():
   
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA 

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED failed to open")
        return

    runtime = sl.RuntimeParameters()
    left_image = sl.Mat()
    depth_map = sl.Mat()
    camera_info = zed.get_camera_information()
    fy = camera_info.calibration_parameters.left_cam.fy  # focal length boyutu (pixel halinde y ekseninde) - odak uzaklığı çekilir 

    
    braking_distance = 6.0 
    stop_distance = 2.0     
'''
aracın durması için en yakın mesafe 2 metre, 6 metreden itibaren frenleme başlıyor.
v^2 = 2ad , a=1.5 m/s^2. Yavaşlama ivmesi 1.5 , hız 4.2 m/s (15 km/h ) olarak belirlenince. 
  '''
    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

                frame = left_image.get_data()

                
                cones = run_yolo(frame)
                largest_cone = get_largest_cone_bbox(cones)

                if largest_cone is not None:
                    
                    dist = get_depth_at_bbox(depth_map, largest_cone)

                    if dist is not None:
                        
                        #Kontrol Mekanizması: 
                        if dist <= stop_distance:
                            fullstop() #FULLSTOP FONK EKLENCEK 
                            
                        elif dist <= braking_distance:
                            breaksystem() #break basış fonk uygulanacak 
                            
                        else:
                            pass #SÜRMEYE DEVAM EDİŞ FONKSİYONU EKLENMELİ
                           

                    else: #mesafeye ulaşılamazsa (dist == None)
                        pass #FALLBACK DERİNLİK HESABI EKLENCEK- LİDAR VE MONOCULAR DEPTH ESTIMATION
                        
                        bbox_height_px = largest_cone['h']
                        dist = fallback_dist(bbox_height_px, real_height_m=0.75, focal_length_py = fy)
                        #KGM Standardı trafik konisi 75 cm boyutunda alınır. fallback hesabı olarak 

                   ''' 
                   visual monitoring için burası, gömülü sistemde gereksiz : 
                    x, y, w, h = largest_cone['x'], largest_cone['y'], largest_cone['w'], largest_cone['h']
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{dist:.2f} m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                      '''
              
                else:
                    pass #Cone yok 
                
                #cv2.imshow("ZED Left View with Cone Detection", frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break

    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



