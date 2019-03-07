import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC


used_pictures = []
used_pictures2 = []
emotions = ["neutral","joy"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
clf = SVC(kernel='linear', probability=True, tol=1e-3) #Set the classifier as a support vector machines with polynomial kernel
data = {}
nr = 0



def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    
    files = glob.glob("GoogleDBFull/%s/*" %emotion)
    
    
    
    
   
    training = files
    return training



def get_landmarks(image):
    

    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
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
        
        
def make_sets():
    training_data = []
    training_labels = []
   
    
    

    for emotion in emotions:
       
       
       
        print(" working on %s" %emotion)
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        
        for item in training:
            
            alreadyExist = 0
            
            try:
                    if len(used_pictures)>0:
                        
                        
                        
                        if item in used_pictures:                       
                            
                            alreadyExist = 1
                        
                        
                    
                        
                        if alreadyExist == 1:
                            continue
                            
                        else:    
                            nr=nr+1
                            image = cv2.imread(item) #open image
                            print(img)
                            used_pictures.append(item)
                            print(nr)

                            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                            for (x,y,w,h) in faces:
                                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                                image = image[y:y+h, x:x+w]



                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                            clahe_image = clahe.apply(gray)
                            get_landmarks(clahe_image)
                            if data['landmarks_vectorised'] == "error":
                                print("no face detected on this one")
                            else:
                                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                                training_labels.append(emotions.index(emotion))
                            
                    else:
                        break

            except:
                pass
                print("ProblemT")
                
                
       
                
    return training_data, training_labels
accur_lin = []


for i in range(0,1):
    try:
        training_data, training_labels = make_sets()
        npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        clf.fit(npar_train, training_labels)
    except:
        print("Error" %nr)    
    
print("Training finished" %nr) #train SVM    
    
