import Tkinter as tk
import numpy as np

#TODO from speaker import test,train
#TODO from face import test,train

class BioLock(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		container = tk.Frame(self)

		container.pack(side="top", fill="both", expand = True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		#TODO Initialize user dictionary
		

		# Initalize new user variables
		self.newAccess = False
		self.newAdmin = False
		self.id = 0
		self.code = np.array()

		# Initalize test variables


		#Initialize Frames
		self.frames = {}

		#TODO Add frames
		for f in [faceCapture,voiceCapture,enroll,enrollFace,enrollVoice,success,failure]:
			frame = f(container,self)
			self.frames[f] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(faceCapture)

	#Brings specified frame to front of app
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()
	
	#
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

class enroll(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enroll new user")
		label.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.facialRecognition)
		button.pack()

class enrollFace(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Face Caputre")
		label.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.facialRecognition)
		button.pack()

class enrollVoice(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Voice Capture")
		label.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.facialRecognition)
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

app = BioLock()
app.mainloop()


