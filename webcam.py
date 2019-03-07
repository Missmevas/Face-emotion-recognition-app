import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from PIL import Image
#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

data = {}



def nparray_as_image(nparray, mode='RGB'):
    """
    Converts numpy's array of image to PIL's Image.
    :param nparray: Numpy's array of image.
    :param mode: Mode of the conversion. Defaults to 'RGB'.
    :return: PIL's Image containing the image.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)

def image_as_nparray(image):
    """
    Converts PIL's Image to numpy's array.
    :param image: PIL's Image object.
    :return: Numpy's array of the image.
    """
    return np.asarray(image)

def draw_with_alpha(source_image, image_to_draw, coordinates):
    
    """
    Draws a partially transparent image over another image.
    :param source_image: Image to draw over.
    :param image_to_draw: Image to draw.
    :param coordinates: Coordinates to draw an image at. Tuple of x, y, width and height.
    """
    
    x, y, w, h = coordinates
    x=int(x)
    y=int(y)
    h=int(h)
    w=int(w)
    
    if x < 0 or y < 0:
        print('cannot display negative coordinates')
        return
    
    #if (x+w) > 
   
    image_to_draw = image_to_draw.resize((int(w),int(h)), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    
    print(coordinates)
    print(source_image.shape)
    print(image_array.shape)
   
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)



        
def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
   
    return [nparray_as_image(cv2.imread('%s.png' % emotion, -1), mode=None) for emotion in emotions]
    


def show_webcam_and_run(emoticons):
   
   
    while not video_capture.isOpened():
        cv2.waitKey(1000)
        print ("Wait for the header")


    while True:


        ret,frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1) #Detect the faces in the image



        for k,d in enumerate(detections): #For each detected face
            shape = predictor(clahe_image, d) #Get coordinates
            xlist = []
            ylist = []
            for i in range(1,68): #There are 68 landmark points on each face

                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1) 

                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
                
            cv2.imshow("image", frame) #Display the frame
            
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xmin = np.min(xlist)
            xmax =np.max(xlist)
            ymin= np.min(ylist)
            ymax= np.max(ylist)
            
            width = abs(xmax-xmin)
            height = abs(ymax-ymin)
            
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]
            landmarks_vectorised = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
            data['landmarks_vectorised'] = landmarks_vectorised
                
        
            if len(detections) < 1:
                data['landmarks_vestorised'] = "error"

            else:
                image_to_draw = emoticons[1]
                draw_with_alpha(frame, image_to_draw, (xmin, ymin, width, height))

                
           
            
        if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
                break





if __name__ == '__main__':
    
    emotions = ["neutral","joy", "surprise","sadness", "anger"]
    emoticons = _load_emoticons(emotions)

    

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(emoticons)