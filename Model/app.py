import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import imutils


class VideoTransformer(VideoTransformerBase):
    fonts =  cv2.FONT_HERSHEY_COMPLEX
    fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fonts4 = cv2.FONT_HERSHEY_TRIPLEX
    x, y , h , w = 0,0 ,0 ,0
    DISTANCE=0
    face_detector = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
    Known_distance =31.5#Inches
    Known_width=5.7#Inches

    #Setting up your arduino
    # #arduino = serial.Serial('/dev/ttyUSB0',9600)

    lowConfidence = 0.75
    GREEN = (0,255,0) 
    RED = (0,0,255)
    BLACK = (0,0,0)
    YELLOW =(0,255,255)
    PURPLE = (255,0,255)
    WHITE = (255,255,255)
    Distance_level =0
    prototxtPath = r"deploy_prototxt.py"
    weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = tf.keras.models.load_model(r"mask_detector.model")

    ref_image = cv2.imread(r"D:\Uni Projects\Final_yarab\Refimage.png")
    #print(ref_image.shape)

    # initialize the video stream
    Motor1_Speed=0 # Speed of motor Accurding to PMW values in Arduino 
    Motor2_Speed=0
    Truing_Speed =110
    net_Speed =180
    
    def FocalLength(self,measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image* measured_distance)/ real_width
        return focal_length
    # distance estimation function
    def Distance_finder (self,Focal_Length, real_face_width, face_width_in_frame):
        # Function Discrption (Doc String)
        #Distance=(Real Face Width)
        distance = (real_face_width * Focal_Length)/face_width_in_frame

        return distance
    #face detectinon function then mask detection function
    def detectAndPredictMask(self,frame, faceNet, maskNet):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > 0.75:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel ordering, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

                # add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        return (locs, preds)

    #calculate the distance of the face and direction
    def face_data(self,image, CallOut, Distance_level):
        # Function Discrption (Doc String)
        '''
        
        This function Detect face and Draw Rectangle and display the distance over Screen
        
        :param1 Image(Mat): simply the frame 
        :param2 Call_Out(bool): If want show Distance and Rectangle on the Screen or not
        :param3 Distance_Level(int): which change the line according the Distance changes(Intractivate)

        :return1  face_width(int): it is width of face in the frame which allow us to calculate the distance and find focal length
        :return2 face(list): length of face and (face paramters)
        :return3 face_center_x: face centroid_x coordinate(x)
        :return4 face_center_y: face centroid_y coordinate(y)
    
        '''
        face_width = 0
        face_x, face_y =0,0
        face_center_x =0
        face_center_y =0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaleFactor=1.3
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        faces = self.face_detector.detectMultiScale(gray_image,  scaleFactor=1.301, minNeighbors=5,minSize=(10, 10)) # better detection at scaling factor 1.21/ conumse more cpu.
        for (x, y, h, w) in faces:
            face_width = w
            face_center=[]
            face_center_x =int(w/2)+x
            face_center_y =int(h/2)+y
            if Distance_level <10:
                Distance_level=10
            if CallOut==True:

                cv2.line(image, (x,y), (face_center_x,face_center_y ), (155,155,155),1)
                cv2.line(image, (x,y-11), (x+210, y-11), (self.YELLOW), 25)
                cv2.line(image, (x,y-11), (x+self.Distance_level, y-11), (self.GREEN), 25)
        return face_width, faces, face_center_x, face_center_y
    # ref_image_face_width,_, _,_= face_data(ref_image, False, Distance_level)
    # Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
    def transform(self,frame):
        
        frame=frame.to_ndarray(format="bgr24")
        check=True
        # grab the frame from the threaded video stream and resize it to have a maximum width of 900 pixels
        frame = imutils.resize(frame, width=900)

        RightBound = 656   
        Left_Bound =280
    # detect faces in the frame and determine if they are wearing a face mask or not
        (locs, preds) = self.detectAndPredictMask(frame, self.faceNet, self.maskNet)

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label =="Mask":
                print("ACCESS GRANTED")
                # #arduino.write(b'H')

            else: 
                print("ACCESS DENIED")
                face_width_in_frame,Faces ,FC_X, FC_Y= self.face_data(frame, True, self.Distance_level)
                for (face_x, face_y, face_w, face_h) in Faces:
                    if face_width_in_frame !=0:
                        Distance = self.Distance_finder(500, self.Known_width,face_width_in_frame)
                        Distance = round(Distance,2)
                        # Drwaing Text on the screen
                        Distance_level= int(Distance)
                        cv2.line(frame, (50,33), (130, 33), (self.BLACK), 15)
                        cv2.putText(frame, f"Robot State", (50,35), self.fonts,0.4, (self.YELLOW),1)
                        # cv2.line(frame, (50,65), (170, 65), (BLACK), 15)
                        
                        # Direction Decider Condition
                        if FC_X<Left_Bound:
                            # Writing The motor Speed
                            check=True 
                            Motor1_Speed=self.Truing_Speed
                            Motor2_Speed=self.Truing_Speed
                            print("Left Movement")
                            # Direction of movement
                            Direction=3
                            cv2.line(frame, (50,65), (170, 65), (self.BLACK), 15)
                            cv2.putText(frame, f"Move Left {FC_X}", (50,70), self.fonts,0.4, (self.YELLOW),1)

                        elif FC_X>RightBound:
                            # Writing The motor Speed 
                            check=True 
                            Motor1_Speed=self.Truing_Speed
                            Motor2_Speed=self.Truing_Speed
                            print("Right Movement")
                            # Direction of movement
                            Direction=4
                            cv2.line(frame, (50,65), (170, 65), (self.BLACK), 15)
                            cv2.putText(frame, f"Move Right {FC_X}", (50,70), self.fonts,0.4, (self.GREEN),1)
                        
                            # cv2.line(frame, (50,65), (170, 65), (BLACK), 15)
                            # cv2.putText(frame, f"Truing = False", (50,70), fonts,0.4, (WHITE),1)

                        elif Distance >70 and Distance<=200:
                            # Writing The motor Speed
                            check=True  
                            Motor1_Speed=self.net_Speed
                            Motor2_Speed=self.net_Speed
                            # Direction of movement
                            Direction=2
                            cv2.line(frame, (50,55), (200, 55), (self.BLACK), 15)
                            cv2.putText(frame, f"Forward Movement", (50,58), self.fonts,0.4, (self.PURPLE),1)
                            print("Move Forward")
                        
                        elif Distance >20 and Distance<=70:
                            # Writing The motor Speed 
                            check=True 
                            Motor1_Speed=self.net_Speed
                            Motor2_Speed=self.net_Speed
                            # Direction of movement
                            Direction=1
                            print("Move Backward")
                            cv2.line(frame, (50,55), (200, 55), (self.BLACK), 15)
                            cv2.putText(frame, f"Backward Movement", (50,58), self.fonts,0.4, (self.PURPLE),1)
                        else:
                            check=True
                        if check ==True:
                            cv2.putText(frame, f"Distance {Distance} Inch", (face_x-6,face_y-6), self.fonts,0.6, (self.BLACK),2)
                            # data = f"A{Motor1_Speed}B{Motor2_Speed}D{Direction}" #A233B233D2
                            
                    # Arduino.write(data.encode()) #Encoding the data into Byte fromat and then sending it to the arduino 
                        time.sleep(0.002) # Providing time to Arduino to Receive data.
                        #Arduino.flushInput() #Flushing out the Input.
                        # # Sending data to Arduino 
                
                # #arduino.write(b'L')
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.line(frame, (Left_Bound, 80), (Left_Bound, 480-80), (self.YELLOW), 2)
        cv2.line(frame, (RightBound, 80),(RightBound, 480-80), (self.YELLOW), 2)
        return frame

        # do a bit of cleanup


def main():
        # Face Analysis Application #
        st.title("Real Time Mask Detection Application")
        activities = ["Home", "Webcam Mask Detection", "About"]
        choice = st.sidebar.selectbox("Select Activity", activities)
        st.sidebar.markdown(
            """ Developed by Kareem Abouelseoud and Fouad Amr """)
        if choice == "Home":
            html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                                <h4 style="color:white;text-align:center;">
                                                Face & Mask detection application using OpenCV, Pre-trained CNN model and Streamlit.</h4>
                                                </div>
                                                </br>"""
            st.markdown(html_temp_home1, unsafe_allow_html=True)
            st.write("""
                    The application has two functionalities.
                    1. Real time face detection using web cam feed.
                    2. Real time face emotion recognization.
                    """)
        elif choice == "Webcam Mask Detection":
            st.header("Webcam Live Feed")
            st.write("Click on start to use webcam and detect your face emotion")
            webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

        elif choice == "About":
            st.subheader("About this app")
            html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Real time Mask detection application using OpenCV, Pre-trained CNN model and Streamlit.</h4>
                                        </div>
                                        </br>"""
            st.markdown(html_temp_about1, unsafe_allow_html=True)

            html_temp4 = """
                                        <div style="background-color:#98AFC7;padding:10px">
                                        <h4 style="color:white;text-align:center;">This Application is developed by Kareem Abouelseoud and Fouad Amr</h4>
                                        <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                        </div>
                                        <br></br>
                                        <br></br>"""

            st.markdown(html_temp4, unsafe_allow_html=True)

        else:
            pass


if __name__ == "__main__":
    main()
