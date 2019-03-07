import os.path
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC


emotions = ["anger", "disgust", "joy", "sadness", "surprise","fear", "neutral","contempt"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []



def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("cohn-kanade/%s/*/*.png" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


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
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)        #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                
                print("no face detected on this one")
            else:
                if emotions.index(emotion) == '':    # CHANGE HERE FOR PROPER LABEL !!!!!
                    print("no label to the picture") 
                else:
                    training_data.append(data['landmarks_vectorised']) #append image array to training data list
                    a1 = []
                    current_emotion = ''
                    with open('Emotions/cohn-kanade/S010/004/S010_004_00000019_emotion.txt') as f:
                        for line in f:
                            data = line.split()
                            a1.append(int(data[0][0]))                           


                            if a1[0]==0:
                            
                                current_emotion="neutral"                             
                              
                            
                            elif a1[0]==1:
                            
                                current_emotion="anger"
                                                                            
                            elif a1[0]==2:
                            
                                current_emotion="contempt"
                                                         
                            
                            elif a1[0]==3:
                            
                                current_emotion="disgust"
                            
                                                       
                            elif a1[0]==4:
                            
                                current_emotion="fear"
                                                         
                            
                            elif a1[0]==5:
                            
                                current_emotion="joy"                               
                             
                            
                            elif a1[0]==6:
                            
                                current_emotion="sadness"
                                                         
                            
                            elif a1[0]==7:
                            
                                current_emotion="surprise"              
                    
                    
                    training_labels.append(current_emotion) # CHANGE HERE FOR PROPER LABEL
                    print("emotions.index(emotion)", current_emotion)
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #NOT ALWAYS NECCESSARY !!!
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                if emotions.index(emotion) == '': 
                    print("no label to the picture") 
                else:    
                    prediction_data.append(data['landmarks_vectorised'])
                    a1 = []
                    current_emotion = ''
                    with open('Emotions/cohn-kanade/S010/004/S010_004_00000019_emotion.txt') as f:    # CHANGE HERE - LABEL !!!!!
                        for line in f:
                            data = line.split()
                            a1.append(int(data[0][0]))                           


                            if a1[0]==0:
                            
                                current_emotion="neutral"                             
                              
                            
                            elif a1[0]==1:
                            
                                current_emotion="anger"
                                                                            
                            elif a1[0]==2:
                            
                                current_emotion="contempt"
                                                         
                            
                            elif a1[0]==3:
                            
                                current_emotion="disgust"
                            
                                                       
                            elif a1[0]==4:
                            
                                current_emotion="fear"
                                                         
                            
                            elif a1[0]==5:
                            
                                current_emotion="joy"                               
                             
                            
                            elif a1[0]==6:
                            
                                current_emotion="sadness"
                                                         
                            
                            elif a1[0]==7:
                            
                                current_emotion="surprise" 
                    
                    
                    
                    prediction_labels.append(current_emotion)              # CHANGE HERE TOO !!!!!!!
    return training_data, training_labels, prediction_data, prediction_labels
accur_lin = []
for i in range(0,1):
    print("Making sets %s" %i) #Make sets by random sampling 80/20
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs




        





        