import Tkinter as tk
import numpy as np
import FinalFaceRecog as fr
#TODO from speaker import test,train

class BioLock(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		#self.attributes('-fullscreen',True)
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand = True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		#TODO Initialize user dictionary (either load from file or start empty one)
		self.users = {}

		# Initalize new user variables
		self.newAccess = False
		self.newAccessString = None
		self.newAdmin = False
		self.newAdminString = None
		self.newID = 0
		self.newCode = None

		# Initalize test variables
		self.access = False
		self.admin = False
		self.id = -1
		self.code = None

		#Initialize Frames
		self.frames = {}

		for f in [faceCapture,voiceCapture,enroll,enrollFace,enrollVoice,success,failure]:
			frame = f(container,self)
			self.frames[f] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		#TODO will depend on intial conditions
		self.show_frame(failure)

	#Brings specified frame to front of app
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()
	
	
	def enrollPriv(self):
		if self.newAccessString.get() == "Yes":
			self.newAccess = True
		if self.newAdminString.get() == "Yes":
			self.newAdmin = True
		self.show_frame(enrollFace)

	def enrollFace(self):
		#Face enrollment code goes here
		#Set self.newID
		self.show_frame(enrollVoice)

	def enrollVoice(self):
		#Voice enrollment code goes here
		#Set self.newCode

		#Save data to map and storage
		self.show_frame(faceCapture)

	def faceAuth(self):
		#Facial Recognition
		#successful=fr.Authenticate() #Add id
		successful = True
		print successful
		if successful:
			#TODO Save id
			self.show_frame(voiceCapture)
		else
			#Reset all variables
			self.show_frame(failure)

	def voiceAuth(self):
		print "voice stuff"
		#TODO Run speaker Recognition (including voice capture)
		success = False
		if success
			self.show_frame(success)
		else
			self.show_frame(failure)

	def unlock(self):
		print "Send Unlock Signal"
		self.show_frame(faceCapture)


#TODO figure out option menu
class enroll(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enroll new user")
		label.pack()
		accessLabel = tk.Label(self, text="Give new user access to system?")
		accessLabel.pack()
		accessMenu = tk.OptionMenu(self,controller.newAccessString,"Yes","No")
		accessMenu.pack()
		adminLabel = tk.Label(self, text="Give new user admin rights")
		adminLabel.pack()
		adminMenu = tk.OptionMenu(self, controller.newAdminString,"No","Yes")
		adminMenu.pack()
		button = tk.Button(self,text = "Next", command=controller.enrollPriv)
		button.pack()

class enrollFace(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Face Caputre")
		label.pack()
		instr = tk.Label(self, text="After pushing button please remain still, facing the camera")
		instr.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.enrollFace)
		button.pack()

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

class faceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Face Image Capture")
		label.pack()
		instr = tk.Label(self, text="After pushing the button please remain still, facing the camera")
		instr.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.faceAuth)
		button.pack()
		

class voiceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Voice Capture")
		label.pack()
		instr = tk.Label(self, text="After pushing the button please say your passphrase")
		button = tk.Button(self, text = "Ready to Record", command=controller.voiceAuth)
		button.pack()


class success(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Authentication Successful")
		label.pack()
		unlock = tk.Button(self, text= "Unlock System", command= controller.unlock)
		unlock.pack()
		enroll = tk.Button(self, text="Enroll new User", command=lambda: controller.show_frame(enroll))
		enroll.pack()


class failure(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Authentication Failed")
		label.pack()
		button = tk.Button(self, text = "OK", command=lambda: controller.show_frame(faceCapture))
		button.pack()

app = BioLock()
app.mainloop()
