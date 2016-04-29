import sys
import os.path

import fnmatch
import cv2
import numpy as np 

import face
import shutil

from picamera import PiCamera
from picamera.array import PiRGBArray
import time


TRAINING_FILE='train_LBPH.xml'
BASE_PATH="training/negative"
cascadePath = "haarcascade_frontalface_alt.xml"
LOOKUP_FILE='lookup_table.txt'
ENROLLMENT_FILE='enrollment.txt'
CSV_FILE='CSV.txt'

def walk_files(directory, match='*'):
	"""Generator function to iterate through all files in a directory recursively
	which match the given filename match parameter.
	"""
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, match):
			yield os.path.join(root, filename)


def prepare_image(filename):
	"""Read an image as grayscale and resize it to the appropriate size for
	training the face recognition model.
	"""
	return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def normalize(X, low, high, dtype=None):
	"""Normalizes a given array in X to a value between low and high."""
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

#----------------------------------------------------------------------------------------------Load LOOKUP TABLE
def load_table(filename,lookup_table,sample_images):
	t=open(filename,'r+')
	line=t.readline()
	#lookup_table=[]
	#sample_images=[]

	while line!="":
	    two=line.split(";")
	    folder_name=two[0]
	    imageName=two[1]
	    lookup_table.append(folder_name)
	    sample_images.append(imageName)
	    #print "folder: "+folder_name+ " !!" +imageName
	    line=t.readline()

#----------------------------------------------------------------------------------------------Create CSV and LOOKUP table
def create_csv():
    #if len(sys.argv) != 2:
        #print "usage: create_csv <base_path>"
        #sys.exit(1)

    SEPARATOR=";"
    lookup_table=[]

    f=open(CSV_FILE,'w')
    t=open(LOOKUP_FILE,'w')

    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            print "!! "+subdirname
            #subject_path = os.path.join(dirname, subdirname)
            subject_path ="%s/%s" % (dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s\%s" % (subject_path, filename)
                # added to create right directorys in linux
                abs_path2="%s/%s" % (subject_path,filename)

                seq=''.join([str(abs_path),str(SEPARATOR),str(label),'\n'])
                #Sprint seq
                f.write(seq)
                #print "%s%s%d" % (abs_path, SEPARATOR, label)
            label = label + 1
            lookup_table.append(subdirname)
            t.write(''.join([str(subdirname),';',abs_path2,';\n']));
    print lookup_table
    # use lookup_table[label] to look up the specific folder of that label
    f.close()
    t.close()

#--------------------------------------------------------------------------------------------TRAIN THE SYSTEM (RUN ONLY ONCE)
def trainLBPH():

	faces = []
	labels = []
	labelnum=[]
	temp=10000
	totalcount=0

	f=open(CSV_FILE,'r+')
	s=f.readline()
	while s!="":
		#print s
		list=s.split(';')
	            #print list
		path=str(list[0]).split('\\')
		#print path[0]
		num=str(list[1]).split('\n')
		if temp!=int(num[0]):
			temp=int(num[0])
			print num[0]
			tempcount=0
			labelnum.append(int(num[0]))			

			for filename in walk_files(path[0],'*.pgm'):
				#print filename
				faces.append(prepare_image(filename))
				labels.append(labelnum[int(num[0])])
				tempcount += 1
				totalcount += 1
			
		else:
			while tempcount > 0:
				s=f.readline()
				tempcount -= 1
				print num[0]+":"+s+"!!"+str(tempcount)+"\n"
				
			continue
	print 'Read', totalcount, 'images'

	#print np.asarray(labels).shape
	#print np.asarray(faces).shape

	#Train model
	print 'Training model...'
	model = cv2.createLBPHFaceRecognizer()
	model.train(np.asarray(faces), np.asarray(labels))

	#Save model results
	model.save(TRAINING_FILE)
	print 'Training data saved to', TRAINING_FILE

#-------------------------------------------------------------------------------------Enroll and update
def LBPHupdate(ID):
	labels=[]
	images=[]
	# make sure this is the right file name
	faceCascade = cv2.CascadeClassifier(cascadePath)
	
	counter=0
	#counter2=0
	foldername=ID;
	if not os.path.exists(foldername):
	    os.makedirs(foldername)

	name=foldername+"/Images"
	camera=PiCamera()
	camera.resolution=(320,240)
	camera.framerate=32
	rawCapture=PiRGBArray(camera,size=(320,240))
	time.sleep(3)

	cv2.namedWindow("Capturing new images")
	camera.capture(rawCapture,format="bgr",use_video_port=True)
	while rawCapture is not None and counter<30:
		image=rawCapture.array
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		result=face.detect_single(gray)
		cv2.imshow("Capturing new images",image)
		if result is None:
			flag=0
			print "could not detect single face. Please retry."
		else:
			x,y,w,h=result
			flag=1
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			scaled_byRatio=face.crop(gray,x,y,w,h)
			resized=face.resize(scaled_byRatio)
			print "Saved captured image No."+str(counter)
			counter=counter+1
			filename = name + str(counter) + ".pgm"
			cv2.imwrite(filename,resized)
	        
		rawCapture.truncate(0)
		camera.capture(rawCapture,format="bgr",use_video_port=True)
		key=cv2.waitKey(1)       

	    	
	camera.close()
	cv2.destroyWindow("Capturing new images")


	#update database
	print 'Loading training data...'
	model=cv2.createLBPHFaceRecognizer()
	model.load(TRAINING_FILE)
	print 'Training data loaded!'

	f=open(CSV_FILE,'r+')
	t=open(LOOKUP_FILE,'r+')
	en=open(ENROLLMENT_FILE,'r+')
	#Get label
	f.seek(-10,2)
	s=f.readline()
	#print s
	list=s.split(';')
	num=str(list[1]).split('\n')
	#new label no.
	label=int(num[0])+1
	#print label

	f.seek(0,2)
	t.seek(0,2)
	en.seek(0,2)

	faces=[]
	labels=[]

	DIRECTORY=foldername
	#print DIRECTORY

	SEPARATOR=";"

	for files in os.listdir(DIRECTORY):
	    abs_path="%s\%s"%(DIRECTORY,files)
	    seq=''.join([str(abs_path),str(SEPARATOR),str(label),'\n'])
	    f.write(seq)
	    
	t.write(''.join([str(DIRECTORY),';',abs_path,';\n']));

	en.write(''.join([str(label),'\n']))

	f.close()
	t.close()
	en.close()

	for filename in walk_files(DIRECTORY,'*.pgm'):
	    #print filename
	    faces.append(prepare_image(filename))
	    labels.append(label)

	model.update(np.asarray(faces), np.asarray(labels))
	#print model

	#Save model results
	model.save(TRAINING_FILE)
	print 'Training data saved to',TRAINING_FILE

	print "successfully updated"

	#shutil.rmtree(foldername)

#------------------------------------------------------------------------------------------
def Authenticate():
	#load lookup table_ ky
	tableName=LOOKUP_FILE
	table=[]
	samples=[]
	load_table(tableName,table,samples)

	# Create window
	cv2.namedWindow("Preview")
	#cv2.namedWindow("Compared")

	# Load training data into model
	print 'Loading training data...'
	model = cv2.createLBPHFaceRecognizer()
	model.load(TRAINING_FILE)
	print 'Training data loaded!'

	confidences=[]
	labels=[]

	camera=PiCamera()
	camera.resolution=(320,240)
	camera.framerate=32
	rawCapture=PiRGBArray(camera,size=(320,240))
	time.sleep(3)

	count=30
	recognition=0
	
	print 'Looking for face...'
	camera.capture(rawCapture,format="bgr",use_video_port=True)
	while rawCapture is not None:
		image=rawCapture.array
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		result=face.detect_single(gray)
		cv2.imshow("Preview",image)
		key=cv2.waitKey(1)
		if result is None:
			print "Please face to the camera "
		else:
			x, y, w, h = result
			# Crop and resize image to face
			crop = face.resize(face.crop(gray, x, y, w, h))
			label, confidence = model.predict(crop)
			confidences.append(confidence)
			labels.append(label)
			cv2.waitKey(1)
			count -= 1
		if count<=0:
			break
		rawCapture.truncate(0)
		camera.capture(rawCapture,format="bgr",use_video_port=True)
		
	print "finish capturing faces"
	camera.close()
	cv2.destroyWindow("Preview")


	temp=[]
	i=0
	length=len(labels)
	while length>0:
		if i==0:
			temp.append(labels[length-1])
			i += 1
			length -= 1
		else:
			tempi=0
			while tempi<i:
				if labels[length-1]!=temp[tempi]:
					tempi += 1
				else:
					length -=1
					break
			if tempi == i:
				temp.append(labels[length-1])
				i += 1
			length -= 1

	print "------LABELS:{}".format(labels)
	print "------DIFFERENT LABELS:{}".format(temp)
	print "------NUMBER OF DIFFERENT LABELS:{}".format(i)

	tempi=0
	numoflabel=0
	if i > 5:
		print "could not enter"
		#print labels
		return 0,-1
	else:
		element=temp[tempi]
		while tempi < i:
			tempj=0
			count=0
			while tempj<len(labels):
				if labels[tempj]==temp[tempi]:
					count += 1
				tempj += 1
			if count > numoflabel :
				numoflabel=count
				element=temp[tempi]
			tempi += 1
		print "element is {}, numoflabel is {}".format(element, numoflabel)


	tempi = 0
	con=0
	while tempi < len(labels):
		if labels[tempi]==element:
			con=con+confidences[tempi]
		tempi += 1
	ave=con/numoflabel

	print "mean of confidences is {}".format(ave)
	print confidences

	# print recognition
	f=open(ENROLLMENT_FILE,'r')
	s=f.readline()
	flag=0
	while s!="":
		index=int(s)
		#print index
		if index==element:
			flag=1
			print "flag TRUE"
			break
		s=f.readline()
	if ave < 50 and flag==1:
		print "authenticated"
		return 1,element
	else:
		print "could not enter"
		return 0,-1

#------------------------------------------------------------------------------------TESTING MAIN

if __name__ == '__main__':
	#------------run this first to train the system
	create_csv()
	trainLBPH()

	#------------enroll new user
	#name="Akin"
	#LBPHupdate(name)

	#------------authenticate
	#successful,label = Authenticate()
	#print successful
	#print label
	
