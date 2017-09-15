import sys
import numpy as np
from scipy import misc

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python Q2_validate.py img_path produced_path")
		exit()

	im_path = sys.argv[1]
	q2_path = sys.argv[2]	
	
	im1 = misc.imread(im_path)	
	im2 = misc.imread(q2_path)

	im_con = np.ndarray.astype(im1/2, dtype=np.int8) 
	diff = im_con - im2

	print(np.sum(diff))
	print(np.sum(diff < 0))
	print(np.sum(diff >= 0))
	print(im1.size)
	print(np.all(diff==0))

	equal = np.array_equal(np.ndarray.astype(im1/2, dtype=np.int8), im2)
	print(equal)

if __name__ == "__main__":
	main()

