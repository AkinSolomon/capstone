# MELFB         Determine matrix for a mel-spaced filterbank
#
# Inputs:       p   number of filters in filterbank
#               n   length of fft
#               fs  sample rate in Hz
#
# Outputs:      x   a (sparse) matrix containing the filterbank amplitudes
#                   size(x) = [p, 1+floor(n/2)]
#
# Usage:        For example, to compute the mel-scale spectrum of a
#               colum-vector signal s, with length n and sample rate fs:
#
#               f = fft(s);
#               m = melfb(p, n, fs);
#               n2 = 1 + floor(n/2);
#               z = m * abs(f(1:n2)).^2;
#
#               z would contain p samples of the desired mel-scale spectrum
def melfb(p, n, fs):

	f0 = 700.0 / fs
	fn2 = int(math.floor(n/2))

	lr = math.log(1 + 0.5/f0) / (p+1);
	# convert to fft bin numbers with 0 for DC term
	x = np.array([0,1,p,p+1])
	bl = n * (f0 * (np.exp(np.array([0, 1, p, p+1]) * lr) - 1))
	b1 = int(math.floor(bl[0]))
	b2 = int(math.ceil(bl[1]))-1
	b3 = int(math.floor(bl[2]))-1
	b4 = int(min(fn2, math.ceil(bl[3])) - 1)-1
	print b1
	print b2
	print b3
	print b4


	pf = np.log(1 + np.array(range(b1,b4),dtype=float)/n/f0) / lr
	fp = np.floor(pf)
	pm = pf - fp


	r = np.concatenate((fp[b2:b4+1], fp[0:b3+1]))
	c = np.concatenate((np.array(range(b2,b4+1)), np.array(range(1,b3+1))))
	v = 2 * np.concatenate((1-pm[b2:b4+1], pm[0:b3+1]))
	print r.shape
	print r
	print c.shape
	print c
	print v.shape
	print v

	#m = sparse(r, c, v, p, 1+fn2);
	m = np.zeros((p,1+fn2))
	for i in v:
		m[r[i],c[i]]=v[i]

	return m
