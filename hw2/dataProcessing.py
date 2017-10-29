import sys
import numpy as np

cat_dict = [ \
		None,
		["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
		None,
		["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
		None,
		["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
		["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
		["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
		["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
		["Female", "Male"],
		None,
		None,
		None,
		["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
]

def cnt_incomplete_data():
	cnt = 0
	with open("train.csv", "r") as inf:
		for line in inf:
			vals = line.strip().split(",")
			vals = [val.replace(" ", "") for val in vals]
			if "?" in vals:
				cnt += 1
		print(cnt)


def write_to_file(in_csv, out_csv, train=True):
	numeric_indices, index = [], 0
	with open(out_csv, "w") as outf:
		for vals in cat_dict:
			if vals is None:
				numeric_indices.append(index)
				index += 1
			else:
				index += len(vals) + 1
		for i, idx in enumerate(numeric_indices):
			outf.write(str(idx))
			if i != len(numeric_indices)-1:
				outf.write(",")
		outf.write("\n")

		with open(in_csv, "r") as inf:
			inf.readline()
			for line in inf:
				vals = line.strip().replace(" ", "").split(",")
				if train is True:
					vals = vals[:-1]

				cnt = 0
				for idx, val in enumerate(vals):
					if cat_dict[idx] is None:
						outf.write( val + ",")
						cnt += 1
					else:
						names = cat_dict[idx]
						vec = [0 for _ in range(len(names)+1)]
						cnt += len(vec)

						if val in names:
							name_idx = names.index(val)
							vec[name_idx] = 1
						else:
							vec[-1] = 1

						for feat_idx, i in enumerate(vec):
							outf.write(str(i))
							if idx != len(vals)-1 or feat_idx != len(vec)-1:
								outf.write(",")

				outf.write("\n")


def build_data():
	in_csv = sys.argv[1]
	out_csv = sys.argv[2]
	train = sys.argv[3]
	if train == "train":
		train = True
	else:
		train = False

	write_to_file(in_csv, out_csv, train=train)


if __name__ == '__main__':
	build_data()
