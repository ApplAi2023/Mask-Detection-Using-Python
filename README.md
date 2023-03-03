# Mask-Detection-Using-Python
This model is created to be used in an autonomous car to distribute masks in a closed environment.

## The model consists of two parts. 
- The Face and Mask Detection model.
- The navigation of the autonomous car depending on the angle and distance the car is from the person.

# Face Detection Model:
1. The first step was to use cv2.CascadeClassifier to to detect faces.
2. Then, if atleast one face was detected and had accuracy greater than 75%, we would begin extracting the range of interest (ROI), converting, resizing, and preprocessing the face image to be ready for the mask detection model.
3. The image of the face is converted to a numpy array and given to the mask detection model. 
4. We used a pretrained-model to detect the masks.

# Navigation:
1. If no mask was detected then a function is called to calculate the distance and angle the car is from closest the person that it detected.

![WhatsApp Image 2023-03-02 at 10 03 17 PM](https://user-images.githubusercontent.com/88090312/222574005-e411d8a1-c81a-4942-8ad6-2325df8ad07c.jpeg)

2. Once the car has reached the person, it will wait 3 seconds for the person to take a mask.
3. Then, the car will reverse and repeat the process all over again.
