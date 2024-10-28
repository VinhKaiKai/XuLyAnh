import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np

#Buoc 1: load ảnh từ kho
path="pic2"
images = []
classNames = []
myList = os.listdir(path)

for nv in myList:

    anhHienTai = cv2.imread(f"{path}/{nv}") #pic2/Donal Trump.jpg
    images.append(anhHienTai)# them nối đuôi các ma trận điểm ảnh 4 bức ảnh này vào
    classNames.append(os.path.splitext(nv)[0]) #tách path ra2 phần, tên là 1 .jpg là 2
    print(nv)
print(len(images)) #4 là 4 ma trạn điểm ảnh
print(classNames)


#Bước 2: Mã hóa
def Mahoa(images): # hàm mã hóa 4 bức ảnh
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển hóa đổi sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("Mã hóa thành công")
print(len(encodeListKnow))


def thamdu(name):
    with open("thamdu.csv","r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(",") # Tách theo dấu phẩy
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datetimestring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{datetimestring}")



#Bước 3: khởi động cammera
cap= cv2.VideoCapture(0) # can change 1 if have two cammera
while True:
    ret, frame= cap.read()
    framS = cv2.resize(frame, (0,0), None, fx=0.5 , fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    #Xác định vị trí khuôn mặt trên ưebcam
    KhungHinhMatHienTai = face_recognition.face_locations(framS) #Lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodeKhungHinhHienTai= face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodeKhungHinhHienTai,KhungHinhMatHienTai):#lấy khuôn mặt và vị trí theo cặp
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)# so sanh
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)#Kiếm cái index nhỏ nhất
        print(matchIndex)



        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            thamdu(name)
        else:
            name = "Unknow"

        #print(in tên lên frame)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)


    cv2.imshow("cua so", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyWindow()