# Importing the required dependencies
import cv2 # for video rendering
import dlib # for face and landmark detection
import imutils # for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist  # to get the landmark ids of the left and right eyes 
from imutils import face_utils
from flask import Flask,request,render_template,redirect,Response,send_file,session
import pickle
import numpy as np
import cv2
import os
import json
from werkzeug.utils import secure_filename
from fileinput import filename
from numpy.linalg import norm
from deepface import DeepFace
import pandas as pd

UPLOAD_FOLDER = 'C:\\Users\\anshg\\Minor Project\\static\\'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detected = ""
done = ""
var=1

# K - Nearest Neighbour
def k_nearest_neighbour(train,test,k=5):
    dist = []
        # No. of images in the Training data and x.shape[1] denotes the corresponding labels
    
    for i in range(train.shape[0]):
        # get the vector and label
        
        ix = train[i,:-1]
        iy = train[i,-1]
        
        # Compute the distance from the test point
        
        d = distance(test,ix)
        dist.append([d,iy])
        
        #sort based upon distance and get top of K
    
    dk = sorted(dist,key=lambda x:x[0])
    
    dk=dk[:k]
        
        #Retreive only the labels
    labels = (np.array(dk))[:,-1]
        #labels = np.array(dk)[:,-1]
        
        #Get frequencies of each label
        
    output = np.unique(labels,return_counts=True)
        
        
        #Find Max. frequency and corresponding label
        
    index = output[1].argmax()
        
    return output[0][index]
        
def distance(x1,x2):
        return np.sqrt(sum((x1-x2)**2))

# defining a function to calculate the EAR
def calculate_EAR(eye):

    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    EAR = (y1+y2) / x1
    return EAR

##############################################################################################

user="abc"

def generate_frames_for_training():
    global var
    cap = cv2.VideoCapture(var)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 
    skip = 0
    face_data = []
    data_path = './data/'
    
    file_name = user

    while True:
        ret,frame = cap.read()

        # If the image is not captured properly(May be the webcam is not started yet)
        if ret == False:
            continue


        cv2.imshow("Frame",frame)
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame,1.3,5)
        faces = sorted(faces, key=lambda f:f[2]*f[3],reverse=True)  #Sorting the faces with w,h parameters

        for face in faces:
            x,y,w,h = face

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            # pick the section of face i.e Region Of Interest. increasing the boundary lines by 10 pixels
            skip+=1
            offset = 10
            x,y,w,h = face
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))

            # Store the image of every 10th face from the video streaming in a file    

            if skip%10 == 0:
                face_data.append(face_section)
                print(len(face_data))

            cv2.imshow("Face_Section",face_section)
            if skip==10:
                path = 'C:\\Users\\anshg\\Minor Project\\Images\\'
                cv2.imwrite(path+file_name+'.png',frame)

        cv2.imshow("Frame",frame)

        key_pressed = cv2.waitKey(1) & 0xFF

        global done
        if key_pressed == ord('q') or skip==200:
            done="Done"
            #converting our face list into a numpy array
            face_data = np.asarray(face_data)
            face_data = face_data.reshape((face_data.shape[0],-1))     # No. of rows = No. of captured faces
            print(face_data.shape)
            #Save this data into file system
            np.save(data_path+file_name+'.npy',face_data)
            print("Data successfully saved at "+data_path+file_name+'.npy')
            return None
    
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
  
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") 

    #Variable store execution state
    first_read = True
    
    global var
    camera=cv2.VideoCapture(var)
    
    skip = 0
    face_data = []
    label =[]

    data_path = './data/'
    path='C:\\Users\\anshg\\Minor Project\\Images\\'

    class_id = 0 #Id for given file

    names ={} #Mapping between names and id

    #Data Preparation

    for fx in os.listdir(data_path):
    
    
        if fx.endswith('.npy'):
            print("Loaded "+fx)
            ## Create a mapping between class_id and the name
            names[class_id] = fx[:-4]
        
            data_item = np.load(data_path+fx)
            face_data.append(data_item)
               
            #Create labels for the class
            target = class_id*np.ones((data_item.shape[0],))
            class_id += 1
            print("Class_id = ",class_id)
            label.append(target)

    face_dataset =np.concatenate(face_data,axis=0)
    face_labels = np.concatenate(label,axis=0).reshape((-1,1))         #It changes the face_labels matrix into a column

    training_dataset = np.concatenate((face_dataset,face_labels),axis=1)   # Concatenating the label as part of the column                                                                          in the joint matrix of x any y

    ############################### Testing ######################################################

    images=0
    flag=False
    Motion_Detected=False
    pred_name='abc'
    
    # Variables
    blink_thresh = 0.50
    succ_frame = 2
    count_frame = 0

    # Eye landmarks
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    # Initializing the Models for Landmark and
    # face Detection
    detector = dlib.get_frontal_face_detector()
    landmark_predict = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    while True:
    
        ret,frame = camera.read()
        # If the video is finished then reset it
        # to the start
        if camera.get(cv2.CAP_PROP_POS_FRAMES) == camera.get(cv2.CAP_PROP_FRAME_COUNT):
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        print(ret)
        if ret==False:
            continue
        
        faces = face_cascade.detectMultiScale(frame,1.3,5)
        images+=1
        frame_ = imutils.resize(frame, width=640)

        # converting frame to gray scale to
        # pass to detector
        img_gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

        #detecting the faces
        faces_ = detector(img_gray)
        count=0
        
        for face in faces:
        
            x,y,w,h = face
            
            # get the face ROI
        
            offset = 10
        
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))
                       
            if flag==False:
                for fx in os.listdir(path):
                    if fx.endswith('.png'):
                        print("Loaded "+fx)
                        img = cv2.imread(path+fx)
                        print("Image Shape= " ,img.shape)
                        print("Frame Shape= ", frame.shape)
                        result=DeepFace.verify(img,frame)
                        print("Same: ",result["verified"])
                        if result["verified"]==True:
                            flag=True    
            print("Images= ",images)
            if Motion_Detected==False:
                for face in faces_:
                    # landmark detection
                    shape = landmark_predict(img_gray, faces_[count])
                    count+=1
                    # converting the shape class directly
                    # to a list of (x,y) coordinates
                    shape = face_utils.shape_to_np(shape)

                    # parsing the landmarks list to extract
                    # lefteye and righteye landmarks--#
                    lefteye = shape[L_start: L_end]
                    righteye = shape[R_start:R_end]

                    # Calculate the EAR
                    left_EAR = calculate_EAR(lefteye)
                    right_EAR = calculate_EAR(righteye)

                    # Avg of left and right eye EAR
                    avg = (left_EAR+right_EAR)/2
                    print("avg=",avg)
                    if avg < blink_thresh:
                        count_frame += 1 # incrementing the frame count
                    else:
                        if count_frame >= succ_frame:
                            Motion_Detected=True
                            print("Motion",Motion_Detected)
                            cv2.putText(frame, 'Blink Detected', (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        else:
                            count_frame = 0
                    
            if Motion_Detected==True:
                cv2.putText(frame, 'Blink Detected', (30, 30),
                               cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                if flag==True:
                    ## Predicted label(out)
                    out = k_nearest_neighbour(training_dataset,face_section.flatten())
                    # display on the screen the name and rectangle around it
                    pred_name = names[int(out)]
                else:
                    pred_name='Intruder'

                cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
                # 1. Image
                # 2. Text_data that you want to write
                # 3. Coordinate where you want the text
                # 4. Type of Font
                # 5. Font scale
                # 6. Color
                # 7. Thickness
                # for better_look line type is "Line_AA"

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                cv2.imshow("Faces",frame)
                global detected
                if images>=100:
                    print("pred_name"+pred_name)
                    print("user"+user)
                    if pred_name==user:
                        detected = "Detected"
                    else:
                        detected = "Not detected"
                    return None
        if Motion_Detected==True:
            cv2.putText(frame, 'Blink Detected', (30, 30),
                               cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
        else:
            cv2.putText(frame, 'No Blink Detected', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
           
        if images>=300:
            detected = "No Motion detected"
            return None
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def hello_world():
    return render_template("signup.html")

##                Database                      ##
df=pd.read_csv("./DataBase.csv")
keys=[];
values=[];
database={};
for user in df["UserName"]:
    keys.append(user)
for password in df["Password"]:
    values.append(password)
for i in range(min(len(keys),len(values))):
    database[keys[i]]=values[i]
##                                              ##

@app.route('/login',methods=['GET'])
def login():
    return render_template("login.html", info="")

@app.route('/video1')
def video1():
    res = generate_frames_for_training()
    return Response(res,mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    res = generate_frames()
    if(res == None):
        return render_template('login.html',info='Invalid User')
    else:
        return Response(res,mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/face_recognition_result')
def face_recognition_result():
    return detected

@app.route('/face_training_completion')
def face_training_completion():
    return done

@app.route('/home')
def home():
    files=[]
    global user
    if not os.path.exists("./static/"+user):
    # if the static_user directory is not present 
    # then create it.
        os.makedirs("static/"+user)
    for i in os.listdir('./static/'+user+'/'):
        if '.' in i:
            files.append(i)
    return render_template('home.html',name=user,file_names=files)

@app.route('/form_signup',methods=['POST','GET'])
def signup():
    selected_webcam = request.form['webcam']
    if selected_webcam == 'external':
        global var
        var=1;
    else:
        var=0
    name1=request.form['username']
    global user
    user=request.form['username']
    password=request.form['password']
    with open("DataBase.csv","a") as file:
        file.write("\n"+user+"," + password)
    return render_template('Camera.html')

@app.route('/form_login',methods=['POST','GET'])
def form_login():
    selected_webcam = request.form['webcam']
    if selected_webcam == 'external':
        global var
        var=1;
    else:
        var=0
    name1=request.form['username']
    global user
    user=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info='Invalid User')
    else:
        if database[name1]!=pwd:
            return render_template('login.html',info='Invalid Password')
        else:
             return render_template('WebCamera.html')

@app.route('/<string:Name>')
def allow(Name):
    global user
    return send_file('static/'+user + '/'+ Name,as_attachment=True)
     
@app.route("/upload",methods=['GET','POST'])
def upload():
    global user
    if(request.method=='POST'):  
        f=request.files.getlist('myfiles')
        for i in f:
            i.save(os.path.join(app.config['UPLOAD_FOLDER']+user+'\\', secure_filename(i.filename)))
    return redirect('/home')

@app.route('/delete/<string:filename>', methods=['GET'])
def delete(filename):
    print("Delete File Path: ")
    print(os.path.join(app.config['UPLOAD_FOLDER']+user+'\\', secure_filename(filename)))
    os.remove(os.path.join(app.config['UPLOAD_FOLDER']+user+'\\', secure_filename(filename)))
    return redirect('/home')

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)