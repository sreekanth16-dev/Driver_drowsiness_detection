import cv2
import tensorflow as tf
import numpy as np
import time
from pygame import mixer




# from twilio.rest import Client

# def make_twilio_call():
#     account_sid = 00
#     auth_token = 00#token numbef
#     twilio_number = 0 # Your Twilio number
#     passenger_number = 0  # The passenger's phone number
#     twiml_app_sid = 0  # Your TwiML App SID

#     client = Client(account_sid, auth_token)

#     # Make a call using the TwiML App SID
#     call = client.calls.create(
#         to=passenger_number,
#         from_=twilio_number,
#         application_sid=twiml_app_sid
#     )

#     print(f"Call initiated! Call SID: {call.sid}")








#installing cascade installer
face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cas=cv2.CascadeClassifier('haarcascade_eye.xml')


#cnn model loading
model=tf.keras.models.load_model('drowsyness_detection1.h5')


#intializing alarm

mixer.init()
alarm=mixer.Sound('severe-warning-alarm-98704.mp3')


#open webcam

video=cv2.VideoCapture(0)
close_eye_count=0
threshold=2


while True:
    success,img=video.read()
    if success ==True:
        gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # color_frame = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        faces=face_cas.detectMultiScale(gray_image,scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        
        for (x,y,w,h) in faces:
            face_reg=gray_image[y:y+h,x:x+w]
            roi_color =img[y:y+h, x:x+w]
            

            eyes=eye_cas.detectMultiScale(face_reg,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            eye_status=[]

            if len(eyes)>0:
                for (ex,ey,ew,eh) in eyes:
                    eye=face_reg[ey:ey+eh,ex:ex+ew] 
                    print(eye)
                    eye=cv2.resize(eye,(64,64,))
                    eye=eye/ 255.0
                    eye=np.expand_dims(eye,axis=-1) #cahnge  dimension to (64,64,1)
                    eye=np.expand_dims(eye,axis=0) #change dimesion to (1,64,64,1)
                    # eye= eye.reshape(1, 64, 64, 1)


                    prediction = model.predict(eye)
                    if prediction[0][0] > 0.5:
                      status = "open"

                    else: 
                        status="close"
                    eye_status.append(status)
                       
                   
                    


                    if status=='open':
                        color=(0,255,0)
                        cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),color,4)
                       
                    elif status=='close':
                         color=(255,0,0)
                         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
                         cv2.putText(img, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if all(status=='close' for status  in eye_status):
                    close_eye_count+=1
                else:
                    close_eye_count=0

                if close_eye_count >= threshold:
                    cv2.putText(img, "ALERT! Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not mixer.get_busy():  # Prevent multiple alarm sounds
                        alarm.play()
                    # 

                      
        cv2.imshow('realtime_view',img)
        
    
    


                                   

            

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()





