import Tkinter as tk
import numpy as np
import sys
import os.path
import xml.etree.ElementTree as ET

import fnmatch
import cv2
import numpy as np

import face
import shutil

import speaker

TRAINING_FILE='train_LBPH.xml'
BASE_PATH="training/negative"
cascadePath = "haarcascade_frontalface_alt.xml"
LOOKUP_FILE='lookup_table.txt'
ENROLLMENT_FILE='enrollment.txt'
CSV_FILE='CSV.txt'


class BioLock(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		# self.attributes('-fullscreen',True)
		container = tk.Frame(self)
		
		container.pack(side="top", fill="both", expand=True)

		container.grid_rowconfigure(0, weight=3)
		container.grid_columnconfigure(0, weight=3)
		
		newcontainer=tk.Frame(self)
		newcontainer.pack(side="bottom",fill="both",expand=True)
		container.grid_rowconfigure(1, weight=7)
		container.grid_columnconfigure(0, weight=7)

		self.text = tk.Text(newcontainer, wrap="word")
		self.text.pack(side="top", fill="both", expand=True)
		self.text.tag_configure("stdout", foreground="#b22222")

		sys.stdout = TextRedirector(self.text, "stdout")
		sys.stderr = TextRedirector(self.text, "stderr")

		############# Here

		self.adminDict = {}
		self.accessDict = {}
		self.codeDict = {}
		tree = ET.parse('voicedata.xml')
		root = tree.getroot()
		for child in root:
			idString = child.tag
			ID = int(idString[1:len(idString)])
			print ID
			self.adminDict[ID] = child.get("admin")
			self.accessDict[ID] = child.get("access")
			self.codeDict[ID] = np.load('Arrays/{}.npy'.format(ID))

		############## To here


		# Initalize new user variables
		self.newAccess = False
		self.newAccessString = None
		self.newAdmin = False
		self.newAdminString = None
		self.newID = -1
		self.newCode = None

		# Initalize test variables
		self.access = False
		self.admin = False
		self.id = -1
		self.code = None

		# Initialize Frames
		self.frames = {}

		
		#######Copy this

		for f in [faceCapture, voiceCapture, enroll, enrollFace, enrollVoice, success, failure,enrollDisclaim,authDisclaim]:
			frame = f(container, self)
			self.frames[f] = frame
			frame.grid(row=0, column=0, sticky="nsew")



		if len(self.codeDict) == 0:
			self.show_frame(enrollDisclaim)
		else:
			self.show_frame(authDisclaim)

		#######	To here

	# Brings specified frame to front of app
	def show_frame(self, cont):
		frame = self.frames[cont]
		frame.tkraise()

	def enrollPriv(self, access, admin):
		if access.get() == "Yes":
			self.newAccess = True
		if admin.get() == "Yes":
			self.newAdmin = True

		self.show_frame(enrollFace)

	def enroll(self):
		name = "Kaylee Ye"

		self.newID = FaceRecognizer.LBPHupdate(name)
		print self.newID
		self.show_frame(enrollVoice)

	def enrollVoice(self):
		# Voice enrollment code goes here

		self.newCode = speaker.train()

		# Save data to dictionary and storage
		self.adminDict[self.newID] = self.newAdmin
		self.accessDict[self.newID] = self.newAccess
		self.codeDict[self.newID] = self.newCode
		self.show_frame(authDisclaim)
		np.save('Arrays/{}'.format(self.newID),self.newCode)
		tree = ET.parse('voicedata.xml')
		root = tree.getroot()
		element = ET.SubElement(root,'s'+str(self.newID),admin=str(self.newAdmin),access=str(self.newAccess))
		tree.write('voicedata.xml')
		


	def faceAuth(self):
		# Facial Recognition
		# successful=fr.Authenticate() #Add id
		successful,label = FaceRecognizer.Authenticate()
		self.id=label

		print successful
		print label
		if successful == 1 and self.id in self.codeDict:
			self.code = self.codeDict[self.id]
			self.show_frame(voiceCapture)
		else:
			self.id = -1
			self.show_frame(failure)

	def voiceAuth(self):
		# Run speaker Recognition (including voice capture)
		successful = speaker.test(self.code)
		if successful:
			self.admin = self.adminDict[self.id]
			self.access = self.accessDict[self.id]
			if self.access:
				self.show_frame(success)
			else:
				self.admin = False
				self.id = -1
				self.code = None
				self.show_frame(failure)
		else:
			self.admin = False
			self.id = -1
			self.code = None
			self.show_frame(failure)

	def unlock(self):
		print "Send Unlock Signal"
		self.access = False
		self.admin = False
		self.id = -1
		self.code = None
		self.show_frame(faceCapture)

	def cancel(self):
		self.newAccess = False
		self.newAccessString = None
		self.newAdmin = False
		self.newAdminString = None
		self.newID = 0
		self.newCode = None
		self.access = False
		self.admin = False
		self.id = -1
		self.code = None
		self.show_frame(authDisclaim)
	
	def newUser(self):
		if self.admin:
			self.show_frame(enrollDisclaim)

	def startEnroll(self):
		self.show_frame(enroll)

	def startAuth(self):
		self.show_frame(faceCapture)

#
# Frames
#

class enrollDisclaim(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text="Enrollment Disclaimer")
		label.pack()
		disclaimer = tk.Label(self, text="This program will collect your biometric information, including a photo of your face as well as your voice. Using this device conveys your acceptance of these terms. Tap OK to continue.")
		disclaimer.pack()
		button = tk.Button(self, text='OK', command=controller.startEnroll)
		button.pack()

class authDisclaim(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text="Disclaimer")
		label.pack()
		disclaimer = tk.Label(self, text="This program will collect your biometric information, including a photo of your face as well as your voice. Using this device conveys your acceptance of these terms. Tap OK to continue.")
		disclaimer.pack()
		button = tk.Button(self, text='OK', command=controller.startAuth)
		button.pack()

class enroll(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		#controller.newAccessString = tk.StringVar(controller)
		#controller.newAccessString.set("Yes")
		#controller.newAdminString = tk.StringVar(controller)
		#controller.newAdminString.set("No")
		access = tk.StringVar(controller)
		access.set("Yes")
		admin = tk.StringVar(controller)
		admin.set("No")
		label = tk.Label(self, text= "Enroll new user")
		label.pack()
		accessLabel = tk.Label(self, text="Give new user access to system?")
		accessLabel.pack()
		#accessMenu = tk.OptionMenu(self,controller.newAccessString,"Yes","No")
		accessMenu = tk.OptionMenu(self,access,"Yes","No")
		accessMenu.pack()
		adminLabel = tk.Label(self, text="Give new user admin rights")
		adminLabel.pack()
		#adminMenu = tk.OptionMenu(self, controller.newAdminString,"Yes","No")
		adminMenu = tk.OptionMenu(self, admin,"Yes","No")
		adminMenu.pack()
		#button = tk.Button(self,text = "Next", command=controller.enrollPriv)
		button = tk.Button(self,text = "Next", command=lambda: controller.enrollPriv(access,admin))
		button.pack()
		cancel = tk.Button(self,text= "Cancel", command=controller.cancel)
		cancel.pack()

class enrollFace(tk.Frame):
	def __init__(self, parent, controller):
		self.controller = controller
		tk.Frame.__init__(self, parent)

		label = tk.Label(self, text="Enrollment Face Capture")
		label.pack()

		toolbar=tk.Frame(self)
		toolbar.pack(side="top",fill="x")
		button = tk.Button(self, text="Capture Facial Data", command=controller.enroll)
		button.pack(in_=toolbar,side="left")



class enrollVoice(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Voice Capture")
		label.pack()
		instr = tk.Label(self, text="After pushing the button please say your passphrase")
		instr.pack()
		button = tk.Button(self, text = "Record Voice Sample", command=controller.enrollVoice)
		button.pack()
		cancel = tk.Button(self,text= "Cancel", command=controller.cancel)
		cancel.pack()

class faceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Face Image Capture")
		label.pack()
		#added new layout
		toolbar = tk.Frame(self)
		toolbar.pack(side="top", fill="x")
		b1 = tk.Button(self, text="Enroll New User", command=controller.enroll)
		b2 = tk.Button(self, text="authenticate", command=controller.faceAuth)
		b3 = tk.Button(self, text="Create csv", command=FaceRecognizer.create_csv)
		b4 = tk.Button(self, text="train", command=FaceRecognizer.trainLBPH)
		#b1.pack(in_=toolbar, side="left")
		b2.pack(in_=toolbar, side="left")
		#b3.pack(in_=toolbar, side="left")
		#b4.pack(in_=toolbar, side="left")
		#self.text = tk.Text(self, wrap="word")
		#controller.text.pack(side="top", fill="both", expand=True)
		#self.text.tag_configure("stdout", foreground="#b22222")

		#sys.stdout = TextRedirector(self.text, "stdout")
		#sys.stderr = TextRedirector(self.text, "stderr")
		#--------------------------------------------------------------------------

class voiceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Voice Capture")
		label.pack()
		instr = tk.Label(self, text="After pushing the button please say your passphrase")
		button = tk.Button(self, text = "Ready to Record", command=controller.voiceAuth)
		button.pack()
		cancel = tk.Button(self,text= "Cancel", command=controller.cancel)
		cancel.pack()


class success(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Authentication Successful")
		label.pack()
		unlock = tk.Button(self, text= "Unlock System", command= controller.unlock)
		unlock.pack()

		enroll = tk.Button(self, text="Enroll new User", command=controller.newUser)

		enroll.pack()


class failure(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Authentication Failed")
		label.pack()
		button = tk.Button(self, text = "OK", command=lambda: controller.show_frame(faceCapture))
		button.pack()

class FaceRecognizer(object):
	# ------------------------------------------------------------------------------------TESTING MAIN

	@classmethod
	def walk_files(cls,directory, match='*'):
		"""Generator function to iterate through all files in a directory recursively
		which match the given filename match parameter.
		"""
		for root, dirs, files in os.walk(directory):
			for filename in fnmatch.filter(files, match):
				yield os.path.join(root, filename)

	@classmethod
	def prepare_image(cls,filename):
		"""Read an image as grayscale and resize it to the appropriate size for
		training the face recognition model.
		"""
		return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

	@classmethod
	def normalize(cls,X, low, high, dtype=None):
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
	@classmethod
	def load_table(cls,filename,lookup_table,sample_images):
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
	@classmethod
	def create_csv(cls):
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
	@classmethod
	def trainLBPH(cls):

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

				for filename in cls.walk_files(path[0],'*.pgm'):
					#print filename
					faces.append(cls.prepare_image(filename))
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
	@classmethod
	def LBPHupdate(cls,ID):
		labels=[]
		images=[]
		# make sure this is the right file name
		faceCascade = cv2.CascadeClassifier(cascadePath)

		cv2.namedWindow("preview")

		vc = cv2.VideoCapture(0) # device ID may not be 0

		counter=0
		#counter2=0
		foldername=ID;
		if not os.path.exists(foldername):
			os.makedirs(foldername)

		name=foldername+"/Images"


		if vc.isOpened(): # try to get the first frame
			rval, frame = vc.read()
			print "opened"
			while rval==False:
				rval,frame=vc.read()
		else:
			print "error"
			rval = False
		# if vc.isOpened(): # try to get the first frame
		#     rval, frame = vc.read()
		#     print "opened"
		# else:
		#     print "error"
		#     rval = False

		while rval and counter<30:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#faces =faceCascade.detectMultiScale(gray)
			result=face.detect_single(gray)
			cv2.imshow("preview",frame)
			#for (x, y, w, h) in face:
			if result is None:
				flag=0
				print "could not detect single face. Please retry."
			else:
				x,y,w,h=result
				#print "hello"
				flag=1
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				scaled_byRatio=face.crop(gray,x,y,w,h)
				resized=face.resize(scaled_byRatio)
				#resized=cv2.resize(gray[y:y+h,x:x+w],(200,200),interpolation=cv2.INTER_CUBIC)
				# keyword "interpolation" should not be left out
				cv2.imshow("preview", frame)
				#print "Ready for capture"
				print "Saved captured image No."+str(counter)
				counter=counter+1
				filename = name + str(counter) + ".pgm"
				cv2.imwrite(filename,resized)

			rval, frame = vc.read()
			key=cv2.waitKey(1)


		vc.release()
		cv2.destroyWindow("preview")


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

		for filename in cls.walk_files(DIRECTORY,'*.pgm'):
			#print filename
			faces.append(cls.prepare_image(filename))
			labels.append(label)

		model.update(np.asarray(faces), np.asarray(labels))
		#print model

		#Save model results
		model.save(TRAINING_FILE)
		print 'Training data saved to',TRAINING_FILE

		print "successfully updated"

		shutil.rmtree(foldername)
		return label
	#------------------------------------------------------------------------------------------
	@classmethod
	def Authenticate(cls):
		#load lookup table_ ky
		tableName=LOOKUP_FILE
		table=[]
		samples=[]
		#self.load_table(tableName,table,samples)

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

		vc = cv2.VideoCapture(0)
		print 'Looking for face...'
		if vc.isOpened(): # try to get the first frame
			rval, frame = vc.read()
			print "opened"
			while rval==False:
				rval,frame=vc.read()
		else:
			print "error"
			rval = False

		count=30
		recognition=0
		while rval:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			result=face.detect_single(gray)
			cv2.imshow("Preview",frame)
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
			rval, frame = vc.read()
		print "finish capturing faces"
		vc.release()
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

		#print labels
		#print temp
		#print i

		tempi=0
		numoflabel=0
		if i > 5:
			print "could not enter"
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
		#print confidences

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

		if ave < 52 and flag==1:
			print "authenticated"
			return 1,element
		else:
			print "could not enter"
			return 0,element

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see(tk.END)
        self.widget.update_idletasks()
        self.widget.configure(state="disabled")


app = BioLock()
app.mainloop()
