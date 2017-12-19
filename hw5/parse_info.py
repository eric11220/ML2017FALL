import numpy as np

# Get year, genres
def vectorize_movies(path, out_path):
	with open(path, "rt", encoding='ISO-8859-1') as inf:
		header = inf.readline()
	
		movie_info, years, genre_set = [], [], set()
		for line in inf:
			idx, title, genres = line.strip().split("::")
			idx = int(idx) - 1
	
			year = title.split('(')[-1].split(')')[0]
			genres = genres.split('|')
			years.append(int(year))
			for genre in genres:
				genre_set.add(genre)

			movie_info.append((idx, year, genres))

	genre_set = list(genre_set)
	years = np.asarray(years)
	mean_year, std_year = np.mean(years), np.std(years)
	with open(out_path, "w") as outf:
		outf.write("idx year")
		for genre in genre_set:
			outf.write(",%s" % genre)
		outf.write("\n")

		for info in movie_info:
			idx, year, genres = info
			movie_vec = np.zeros((1+len(genre_set),))
			movie_vec[0] = (int(year) - mean_year) / std_year
			for genre in genres:
				movie_vec[genre_set.index(genre)+1] = 1

			outf.write(str(idx))
			for val in movie_vec:
				outf.write("," + str(val))
			outf.write("\n")


def vectorize(val, val_list):
	vec = np.zeros((len(val_list),))
	if type(val) == list:
		for v in val:
			vec[val_list.index(v)] = 1
	else:
		vec[val_list.index(val)] = 1
	return vec


def vectorize_users(path, out_path):
	with open(path, "rt", encoding='ISO-8859-1') as inf:
		header = inf.readline()

		user_info, ages, gender_set, job_set, zip_set = [], [], set(), set(), set()
		for line in inf:
			idx, gender, age, job, zipcode = line.strip().split("::")
			zipcode = zipcode[0]
			idx, job, zipcode = int(idx)-1, int(job), int(zipcode)
			user_info.append((idx, gender, age, job, zipcode))

			ages.append(age)
			gender_set.add(gender)
			job_set.add(job)
			zip_set.add(zipcode)

	ages = np.asarray(ages, dtype=int)
	mean_age, std_age = np.mean(ages), np.std(ages)
	gender_set, job_set, zip_set = list(gender_set), list(job_set), list(zip_set)
	with open(out_path, "w") as outf:
		outf.write("idx age")
		for gender in gender_set:
			outf.write(",gender_" + gender)
		for job in job_set:
			outf.write(",job_%d" % job)
		for zipcode in zip_set:
			outf.write(",zipcode_%d" % zipcode)
		outf.write("\n")

		user_info.sort(key=lambda tup:tup[0])
		for info in user_info:
			idx, gender, age, job, zipcode = info
			user_vec = np.concatenate((np.zeros((1,)),
																vectorize(gender, gender_set),
																vectorize(job, job_set),
																vectorize(zipcode, zip_set)))
			user_vec[0] = (int(age) - mean_age) / std_age

			outf.write(str(idx))
			for val in user_vec:
				outf.write("," + str(val))
			outf.write("\n")
	

def main():
	vectorize_movies("data/movies.csv", "data/movie_vector.csv")
	vectorize_users("data/users.csv", "data/user_vector.csv")


if __name__ == '__main__':
	main()
