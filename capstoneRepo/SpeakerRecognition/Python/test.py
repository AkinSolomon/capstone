from speaker import test,train

c = train("data/train/s9.wav")
print test("data/test/s9.wav",c)
