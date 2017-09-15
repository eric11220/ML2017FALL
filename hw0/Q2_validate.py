import sys
from PIL import Image

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python Q2_validate.py img_path produced_path")
		exit()

	im_path = sys.argv[1]
	q2_path = sys.argv[2]	

	im1 = Image.open(im_path)
	im2 = Image.open(q2_path)

	im1_pixel = im1.load()
	im2_pixel = im2.load()

	print("original image size: " + str(im1.size))
	print("modified image size: " + str(im2.size))
	input("")

	for i in range(im1.size[0]):
		for j in range(im1.size[1]):
			r1, g1, b1 = im1_pixel[i, j]
			r2, g2, b2 = im2_pixel[i, j]
			if int(r1/2) != r2 or int(g1/2) != g2 or int(b1/2) != b2:
				print("Value unmatched -- im1: " + str(im1_pixel[i, j]) + ", im2: " + str(im2_pixel[i, j]))

if __name__ == "__main__":
	main()
