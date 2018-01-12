import numpy as np


def load_img(path="data/image.npy"):
	return np.load(path)


def load_question(path="data/test_case.csv"):
	questions = []
	with open(path, "r") as inf:
		header = inf.readline()
		print(header)
		for line in inf:
			idx, img1, img2 = line.strip().split(",")
			questions.append((int(img1), int(img2)))
	return questions


def answer_question(questions, cluster_info):
	answers = []
	for q in questions:
		idx1, idx2 = q
		cluster1, cluster2 = cluster_info[idx1], cluster_info[idx2]
		answers.append(cluster1 == cluster2)
	return np.asarray(answers, dtype=int)


def load_correct_answers(questions, path="data/label.txt"):
	truths = []
	with open(path, "r") as inf:
		for line in inf:
			truths.append(line.strip())

	correct_ans = []
	for q in questions:
		idx1, idx2 = q
		cluster1, cluster2 = truths[idx1], truths[idx2]
		correct_ans.append(cluster1 == cluster2) 
	return np.asarray(correct_ans, dtype=int)


def compute_f1(answers, correct_answers):
	tp, fp, fn = 0, 0, 0
	for pred, y in zip(answers, correct_answers):
		if pred == 1 and y == 1:
			tp += 1
		elif pred == 1 and y == 0:
			fp += 1
		elif pred == 0 and y == 1:
			fn += 1

	p, r = tp / (tp+fp), tp / (tp+fn)
	f1 = 2*p*r / (p+r)
	return f1


def write_ans_to_file(answers, path="results/test.csv"):
	with open(path, 'w') as outf:
		outf.write("ID,Ans\n")
		for idx, ans in enumerate(answers):
			outf.write("%d,%d\n" % (idx, ans))
