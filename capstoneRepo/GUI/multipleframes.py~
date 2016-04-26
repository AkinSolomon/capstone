#TODO complete frame design
# Framework based on https://pythonprogramming.net/change-show-new-frame-tkinter/

import Tkinter as tk
import numpy as np

#TODO from speaker import test,train
#TODO from face import test,train

class BioLock(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.attributes('-fullscreen',True)
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand = True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		#TODO Initialize user dictionary (either load from file or start empty one)
		

		# Initalize new user variables
		self.newAccess = False
		self.newAccessString = ""
		self.newAdmin = False
		self.newAdminString = ""
		self.newID = 0
		self.newCode = None

		# Initalize test variables


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
		if self.newAccessString == "Yes":
			self.newAccess = True
		if self.newAdminString == "Yes":
			self.newAdmin = True
		self.show_frame(enrollFace)

	def faceAuth(self):
		#Run Facial Recognition (including image capture)
		print "Face Stuff"

		self.show_frame(voiceCapture)

	def voiceAuth(self):
		#Run speaker Recognition (including voice capture)
		print "voice stuff"
		self.show_frame(success)
	
	def enroll(self):
		self.show_frame(enroll)	

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
		button = tk.Button(self, text = "Capture Facial Data", command=controller.faceAuth)
		button.pack()

class enrollVoice(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Voice Capture")
		label.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.faceAuth)
		button.pack()

class faceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Face Image Capture")
		label.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.faceAuth)
		button.pack()
		

class voiceCapture(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Voice Capture")
		label.pack()
		button = tk.Button(self, text = "Ready to Record", command=controller.voiceAuth)
		button.pack()


class success(tk.Frame):
	def __init__(self,parent,controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Authentication Successful")
		label.pack()
		unlock = tk.Button(self, text= "Unlock System", command= controller.unlock)
		unlock.pack()
		enroll = tk.Button(self, text="Enroll new User", command= controller.enroll)
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


