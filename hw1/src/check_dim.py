import os
import sys

def main():
	csv = sys.argv[1]	
	
	dim = None
	with open(csv, 'r') as inf:
		cnt = 0
		for line in inf:
			vals = line.strip().split(' ')
			if dim is None:
				dim = len(vals)

			if dim != len(vals):
				print(cnt, dim, len(vals))

			cnt += 1
	print(dim)
			
if __name__ == '__main__':
	main()
