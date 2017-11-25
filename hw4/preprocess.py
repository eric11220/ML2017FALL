import os
import re
import sys
PROCESSED_DIR = "processed"

def contain_http(string):
	return "http" in string


def is_ascii(string):
	return all(ord(c) < 128 for c in string)


def main():
	argc = len(sys.argv)
	if argc != 3:
		print("usage: python preprocess.py data_file label_or_not")
		exit()

	in_file = sys.argv[1]
	labeled = sys.argv[2] == '1'

	name, ext = os.path.splitext(in_file)
	out_file = os.path.join(PROCESSED_DIR, in_file)

	if labeled is True:
		label_fp = open(os.path.join(PROCESSED_DIR, name + '_label' + ext), 'w')
	
	cnt = 0
	with open(in_file, "r") as inf:
		with open(os.path.join(out_file), "w") as outf:
			for line in inf:
				# remove any string containing strange "http" specifically for this dataset
				if contain_http(line):
					continue	
	
				line = line.strip().split(' ')
				if labeled is True:
					label, separator, elements = line[0], line[1], line[2:]
					label_fp.write(label + '\n')
				else:
					elements = line
	
				for element in elements:
					# at least one character not in ascii
					if not is_ascii(element):
						continue
	
					# all characters are alphanumeric
					if re.match('\w', element) is not None:
						outf.write(element + ' ')
				outf.write("\n")


if __name__ == '__main__':
	main()
