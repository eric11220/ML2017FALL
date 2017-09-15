import sys
from PIL import Image

def main():
	argc = len(sys.argv)
	if argc != 2:
		print("Usage: python q2.py img_path")
		exit()

	img_path = sys.argv[1]
	img = Image.open(img_path)
	pixels = img.load()

	for i in range(img.size[0]):
		for j in range(img.size[1]):
			r, g, b = pixels[i, j]
			pixels[i, j] = (int(r/2), int(g/2), int(b/2))

	img.save("Q2.jpg", format='PNG')

	'''
	im2 = Image.open("Q2.jpg")
	pixels2 = im2.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			print(str(pixels[i, j]) + " " + str(pixels2[i, j]))
			input("")
	'''

if __name__ == '__main__':
	main()