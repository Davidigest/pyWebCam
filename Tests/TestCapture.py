import cv2
import cv2.cv as cv
import webbrowser
from cv2.cv import *

imageSize = \
[
    (160,   120),
    (320,   240),
    (424,   240),
    (640,   360),
    (800,   448),
    (960,   544),
    (1280,  768),
    (1920 , 1080),
    (3840 , 2160)
]

title='camera'

cv.NamedWindow(title, cv.CV_WINDOW_AUTOSIZE)
capture = cv.CaptureFromCAM(0)

cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT,360)

cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FORMAT, cv.IPL_DEPTH_32F)

# print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
# print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)


cascade = cv.Load(r'haarcascade_frontalface_alt.xml')    

while True:
    img = cv.QueryFrame(capture)          

    storage = cv.CreateMemStorage(0)    
    faces = cv.HaarDetectObjects(img, cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, (100,100))

    for (x, y, w, h), n in faces:
        cv.Rectangle(img, (x, y), (x + w,y + h), 255)
    
    cv.ShowImage(title, img)
    key = cv.WaitKey(100)

    if  key == 32 :                         # Space
        cv.SaveImage("captured.png", img)
        webbrowser.open_new_tab("captured.png")        
        
    elif key == 27:                # Escape    	
        break

cv.DestroyWindow(title)