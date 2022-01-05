import numpy as np
#*******************************Hamming***********************************************#
def HammingDist(template1, mask1, template2, mask2):
    #Here we use hamming distance and put into consideration the masks
    #!(or(mask1,mask2))=and(mask1,mask2)
    #HD=and(xor(temp1,temp2),mask1&mask2)
    # what we do we xor the corresponding bits of temp 1 and temp 2
    # so we will get one only if the two bits are different
    #then we and them with the masks to make sure that the bit is not a noise 
	hd = np.nan
	# Shift template left and right, use the lowest Hamming distance
	for shifts in range(-8,9):
		template1s = shiftbits(template1, shifts)
		mask1s = shiftbits(mask1, shifts)
		mask = np.logical_or(mask1s, mask2)
		nummaskbits = np.sum(mask==1)
		totalbits = template1s.size - nummaskbits
		C = np.logical_xor(template1s, template2)
		C = np.logical_and(C, np.logical_not(mask))
        # we calc the number of 1s in C as it is the number of different bits
		bitsdiff = np.sum(C==1)

		if totalbits==0:
			hd = np.nan
		else:
            #to get the percentage of different bits we divide the number of different bits by the total number of bits
			hd1 = bitsdiff / totalbits
			if hd1 < hd or np.isnan(hd):
				hd = hd1

	# Return
	return hd
def shiftbits(template, noshifts):
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	
	return templatenew
#******************************************************************************#