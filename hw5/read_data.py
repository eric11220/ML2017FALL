import numpy as np

def read_train_data(path):
	max_uid, max_mid, user_movie, ratings = 0, 0, [], []
	with open(path, 'r') as inf:
		header = inf.readline()
		for line in inf:
			_, uid, mid, rating = line.strip().split(',')
			uid = int(uid) - 1
			mid = int(mid) - 1

			user_movie.append([uid, mid])
			ratings.append(rating)

			if uid > max_uid:
				max_uid = uid
			if mid > max_mid:
				max_mid = mid

	return np.asarray(user_movie), np.asarray(ratings, dtype=np.float32), max_uid+1, max_mid+1


def random_shuffle(in_path, out_path, kfold=10):
	import random
	user_ratings = {}
	with open(in_path, 'r') as inf:
		header = inf.readline()
		for line in inf:
			_, uid, mid, rating = line.strip().split(',')
			if user_ratings.get(uid, None) is None:
				user_ratings[uid] = [(mid, rating)]
			else:
				user_ratings[uid].append((mid, rating))

	for uid in user_ratings.keys():
		random.shuffle(user_ratings[uid])

	idx = 0
	with open(out_path, 'w') as outf:
		outf.write(header)
		for fold in range(kfold):
			for uid , ratings in user_ratings.items():
				start = len(ratings) // kfold * fold
				if fold == kfold-1:
					end = len(ratings)
				else:
					end = len(ratings) // kfold * (fold+1)

				for mid, rating in ratings[start:end]:
					outf.write("%d,%s,%s,%s\n" % (idx, uid, mid, rating))
					idx += 1


def get_all_user_movie(path):
	user_idx, movie_idx = set(), set()
	with open(path, 'r') as inf:
		inf.readline()
		for line in inf:
			line = line.strip().split(',')
			uid, mid = line[1], line[2]
			user_idx.add(uid)
			movie_idx.add(mid)

	return user_idx, movie_idx


def main():
	'''
	random_shuffle("data/train.csv", "data/train_shuf.csv")
	train_uids, train_mids = get_all_user_movie("data/1.csv")
	test_uids, test_mids = get_all_user_movie("data/2.csv")
	print("UID:")
	for uid in test_uids:
		if uid not in train_uids:
			print(uid)

	print("MID:")
	for mid in test_mids:
		if mid not in train_mids:
			print(mid)
	'''


if __name__ == '__main__':
	main()