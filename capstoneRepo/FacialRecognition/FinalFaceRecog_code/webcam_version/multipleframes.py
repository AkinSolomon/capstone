
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
		self.adminDict = {}
		self.accessDict = {}
		self.codeDict = {}

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

		#Initialize Frames
		self.frames = {}

		for f in [faceCapture,voiceCapture,enroll,enrollFace,enrollVoice,success,failure]:
			frame = f(container,self)
			self.frames[f] = frame
			frame.grid(row=0, column=0, sticky="nsew")

		#TODO will depend on intial conditions
		self.show_frame(enroll)


	#Brings specified frame to front of app
	def show_frame(self,cont):
		frame = self.frames[cont]
		frame.tkraise()
	
	
	def enrollPriv(self,access,admin):
		if access.get() == "Yes":
			self.newAccess = True
		if admin.get() == "Yes":
			self.newAdmin = True
		print access.get()
		print admin.get()
		print self.newAccess
		print self.newAdmin
		self.show_frame(enrollFace)

	def enrollFace(self):
		#Face enrollment code goes here
		#self.newID = train()
		self.show_frame(enrollVoice)

	def enrollVoice(self):
		#Voice enrollment code goes here
		#self.newCode = train()

		#Save data to dictionary and storage
		self.adminDict[self.newID]=self.newAdmin
		self.accessDict[self.newID]=self.newAccess
		self.codeDict[self.newID]=self.newCode
		self.show_frame(faceCapture)

	def faceAuth(self):
		#Facial Recognition
		#successful=fr.Authenticate() #Add id
		successful = True
		if successful:
			#TODO Find code,admin,access
			self.code = self.codeDict[self.id]
			self.show_frame(voiceCapture)
		else:
			self.id = -1
			self.show_frame(failure)

	def voiceAuth(self):
		#TODO Run speaker Recognition (including voice capture)
		#success = test(self.code)
		success = True
		if success:
			self.admin = self.adminDict[self.id]
			self.access = self.accessDict[self.id]
			self.show_frame(success)
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
		self.show_frame(faceCapture)


#TODO figure out option menu
class enroll(tk.Frame):
	def __init__(self,parent,controller):
		self.controller = controller
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
	def __init__(self,parent,controller):
		self.controller = controller
		tk.Frame.__init__(self,parent)
		label = tk.Label(self, text= "Enrollment Face Caputre")
		label.pack()
		instr = tk.Label(self, text="After pushing button please remain still, facing the camera")
		instr.pack()
		button = tk.Button(self, text = "Capture Facial Data", command=controller.enrollFace)
		button.pack()
		cancel = tk.Button(self,text= "Cancel", command=controller.cancel)
		cancel.pack()

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
		cancel = tk.Button(self,text= "Cancel", command=controller.cancel)
		cancel.pack()


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
