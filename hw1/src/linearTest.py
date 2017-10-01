import sys

def parse_csv(path):
	with open(path, 'r') as inf:
		for line in inf:
			pass

def main():
	argc = len(sys.argv)
	if argc != 3:
		print("Usage: python linearReg.py input_csv output_csv")
		exit()

	in_csv = sys.argv[1]
	parse_csv(in_csv)

if __name__ == '__main__':
	main()
