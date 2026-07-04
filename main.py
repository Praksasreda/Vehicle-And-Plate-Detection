import cv2
import threading
from ultralytics import YOLO
import time
import easyocr

class Kamera:
    def __init__(self,index):
        self.index = index
        self.frame = None
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print(f"Napaka: kamera {index} ni na voljo")

        self.thread = threading.Thread(target=self._update , daemon= True)
        self.thread.start()

    def _update(self):
        while True:
            if self.cap.isOpened():
                ret,frame = self.cap.read()
                if ret:
                    self.frame = frame
            time.sleep(0.01)

    def release(self):
        self.cap.release()



def main():
    model = YOLO("yolov8n.pt")
    model_tablice = YOLO("license_plate_detector.pt")
    kamera1 = Kamera(1)

    reader = easyocr.Reader(['en'])
    time.sleep(1.0)


    while True:
        trenutni_okvir = kamera1.frame
        if trenutni_okvir is not None:
            results = model(trenutni_okvir,verbose = False)
            
            
            for box in results[0].boxes:
                razredId = int(box.cls[0])
                imeRazred = model.names[razredId]
                if razredId == 2:
                    x1,y1,x2,y2 = ([int(e) for e in (box.xyxy[0])])
                    sirina_boxa = x2 - x1
                    sirina_slike = trenutni_okvir.shape[1]

                    if sirina_boxa > sirina_slike * 0.3:

                        izrez = trenutni_okvir[y1:y2,x1:x2]
                        izrez_results = model_tablice(izrez,verbose=False)
                        izrez_box = izrez_results[0].boxes
                        if len(izrez_box) >0:
                            tx1,ty1,tx2,ty2 = [int(e) for e in izrez_box.xyxy[0]]
                            izrezRegisterske = izrez[ty1:ty2,tx1:tx2]
                            branje_izreza = reader.readtext(izrezRegisterske,allowlist="ABCDEFGHIJKLMNOPQRSTVUWZXY0123456789")
                            
                            for detection in branje_izreza:
                                tekst = detection[1]
                                zaupanje = detection[2]
                                if zaupanje>=0.95 and len(tekst)==7:
                                    print(f"tekst:{tekst}, zaupanje {zaupanje}")

                            cv2.rectangle(trenutni_okvir,(x1+tx1,y1+ty1),(x1+tx2,y1+ty2),(255,0,0),2)
                        
                    else :
                        print("ni dovolj blizu")
            
                    cv2.rectangle(trenutni_okvir,(x1,y1),(x2,y2),(0,0,255),2)

                        
            cv2.imshow("prikaz kamere",trenutni_okvir)
        
            



        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    
    print("zapiram kamero")
    kamera1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
