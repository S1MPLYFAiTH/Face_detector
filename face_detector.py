import cv2

#load The Cascade
face_cascade=cv2.CascadeClassifier(r"D:\Machine Learning A-Z Template Folder\haarcascade_frontalface_alt.xml")# Enter the path of haarcascade.xml

#Read the input
vid=cv2.VideoCapture(r"D:\Machine Learning A-Z Template Folder\test_video")# 0 for web camera

ret, frame = vid.read()
#ret -it is a boolean variable that returns true if the frame is available
#frame -it is an image array captured based on the frames per second

while vid.isOpened() and ret:   #to open the purticular video
    ret, frame=vid.read()
    frame=cv2.resize(frame,(1080,720))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces=face_cascade.detectMultiScale(gray)
    
    #Draw rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #Display 
    cv2.imshow("img",frame) 
    key=cv2.waitKey(30)
    if key == ord('q'): #Getting the ascii value of 'q'
        break
    
    
vid.release()  #to release the video or else it will run infintely
cv2.destroyAllWindows()
    
    
