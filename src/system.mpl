sigma:=[
	x0**4 + x1**4 + x2**7 + x3**2 + x4**5 + x5**5 + x6**5,
	x0**4*x1**4 + x0**4*x6**5 + x1**4*x2**7 + x2**7*x3**2 + x3**2*x4**5 + x4**5*x5**5 + x5**5*x6**5,
	x0**4*x1**4*x2**7 + x0**4*x1**4*x6**5 + x0**4*x5**5*x6**5 + x1**4*x2**7*x3**2 + x2**7*x3**2*x4**5 + x3**2*x4**5*x5**5 + x4**5*x5**5*x6**5,
	x0**4*x1**4*x2**7*x3**2 + x0**4*x1**4*x2**7*x6**5 + x0**4*x1**4*x5**5*x6**5 + x0**4*x4**5*x5**5*x6**5 + x1**4*x2**7*x3**2*x4**5 + x2**7*x3**2*x4**5*x5**5 + x3**2*x4**5*x5**5*x6**5,
	x0**4*x1**4*x2**7*x3**2*x4**5 + x0**4*x1**4*x2**7*x3**2*x6**5 + x0**4*x1**4*x2**7*x5**5*x6**5 + x0**4*x1**4*x4**5*x5**5*x6**5 + x0**4*x3**2*x4**5*x5**5*x6**5 + x1**4*x2**7*x3**2*x4**5*x5**5 + x2**7*x3**2*x4**5*x5**5*x6**5,
	x0**4*x1**4*x2**7*x3**2*x4**5*x5**5 + x0**4*x1**4*x2**7*x3**2*x4**5*x6**5 + x0**4*x1**4*x2**7*x3**2*x5**5*x6**5 + x0**4*x1**4*x2**7*x4**5*x5**5*x6**5 + x0**4*x1**4*x3**2*x4**5*x5**5*x6**5 + x0**4*x2**7*x3**2*x4**5*x5**5*x6**5 + x1**4*x2**7*x3**2*x4**5*x5**5*x6**5,
	x0**4*x1**4*x2**7*x3**2*x4**5*x5**5*x6**5 - 1
]:
vars:=[x0,x1,x2,x3,x4,x5,x6]:
start:=time():
gb:=Groebner[Basis](sigma, tdeg(op(vars))):
finish:=time()-start:
writeto("outputFile"):
printf(`Time %f`, finish):