import os
import argparse
import numpy as np
np.set_printoptions(suppress=True)

from skimage import io

reconst_dir = "reconst_faces"


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("--eigen_face", help="Eigen faces path", default="eigen_face.npz")
	parser.add_argument("--imgs_path", help="Images path")
	parser.add_argument("--target_img", help="Target image path")

	return parser.parse_args()


# imgs shape: N * feat
def pca(imgs, k=4):
	u, s, v = np.linalg.svd(imgs, full_matrices=False)
	print(u.shape, s.shape, v.shape)
	return v[:k, :], s


def read_all_imgs(path="Aberdeen"):
	all_imgs = []
	num_imgs = len(os.listdir(path))

	for f in range(num_imgs):
		f = os.path.join(path, str(f) + ".jpg")
		img = io.imread(f)
		img = img.flatten()
		all_imgs.append(img)

	all_imgs = np.asarray(all_imgs, dtype=np.float32)
	return all_imgs


def plot_eigens(channel_eigens):	
	num_face, num_feats  = channel_eigens.shape
	width = int(np.sqrt(num_feats/3))

	for fid in range(num_face):
		face = np.array(channel_eigens[fid, :])
		face -= np.min(face)
		face /= np.max(face)

		face = np.reshape(face, (width, width, 3))
		face = (face * 255).astype(np.uint8)
		path = "eigen_face%d.jpg" % fid
		io.imsave(path, face)


def reconstruct_faces(face, channel_eigens, mean, num_eigen=4):
	face = face.astype(np.float32)
	reconst = np.dot( np.dot(face, channel_eigens[:num_eigen, :].T), channel_eigens[:num_eigen, :])
	reconst += mean
	reconst -= np.min(reconst)
	reconst /= np.max(reconst)
	reconst = (reconst * 255).astype(np.uint8)
	return reconst


def main():
	args = parse_input()

	if os.path.isfile(args.eigen_face):
		arr = np.load(args.eigen_face)
		channel_eigens, ratios, mean = arr['arr_0'], arr['arr_1'], arr['arr_2']
	else:
		all_imgs = read_all_imgs(path=args.imgs_path)
		mean = np.mean(all_imgs, axis=0)
		all_imgs -= mean
		channel_eigens, ratios = pca(all_imgs)
		#np.savez(args.eigen_face, channel_eigens, ratios, mean)

	img_path = os.path.join(args.imgs_path, args.target_img)

	img = io.imread(img_path)
	img = img.flatten()
	img = img - mean
	reconst_face = reconstruct_faces(img, channel_eigens, mean)
	reconst_face = np.reshape(reconst_face, (600, 600, 3))
	io.imsave("reconstruction.jpg", reconst_face)

	'''
	plot_eigens(channel_eigens)
	face_indices = np.asarray([0, 15, 178, 287])
	reconst_faces = reconstruct_faces(all_imgs[face_indices], channel_eigens, mean)
	print(ratios[:4] / np.sum(ratios))
	plot_eigens(channel_eigens)

	if not os.path.isdir(reconst_dir):
		os.makedirs(reconst_dir, exist_ok=True)

	num_face, num_feats  = channel_eigens.shape
	width = int(np.sqrt(num_feats/3))
	for face, idx in zip(reconst_faces, face_indices):
		face = np.reshape(face, (width, width, 3))
		orig_face = np.reshape(all_imgs[idx, :] + mean, (width, width, 3))

		path = "reconst_face%d.jpg" % idx
		path = os.path.join(reconst_dir, path)
		io.imsave(path, face)

		path = "orig_face%d.jpg" % idx
		path = os.path.join(reconst_dir, path)
		io.imsave(path, orig_face.astype(np.uint8))
	'''


if __name__ == '__main__':
	main()
