import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


class Solver:
	def __init__(self):
		self.A = None
		self.LJ = []
		self.LI = []
		self.size = ()
		self.m = None

	def to_csr(self, m: np.array):
		a_index = 0
		self.m = m
		self.size = m.shape
		self.A = m[m != 0]
		self.LJ = np.nonzero(m)[1]
		for ri in range(m.shape[0]):
			non_zero_columns = np.where(m[ri] != 0)[0]
			if len(non_zero_columns):
				self.LI.append(a_index)
				a_index += len(non_zero_columns)
			else:
				self.LI.append(None)
		li_len = len(self.LI)
		for i in range(li_len):
			if self.LI[li_len - i - 1] is None:
				self.LI[li_len - i - 1] = self.LI[li_len - i]
		self.LI.append(len(self.A) + 1)

	def csr_from_file(self, file_name: str):
		data = open(file_name, 'r')
		_ = data.readline()
		n_rows, n_cols, n_elements = data.readline().split()
		print(n_rows, n_cols, n_elements)
		matrix = np.zeros((int(n_rows), int(n_cols)), dtype='float64')
		for pos, line in enumerate(data.readlines()):
			row_index, col_index, element = line.split()
			matrix[int(row_index) - 1, int(col_index) - 1] = float(element)
			matrix[int(col_index) - 1, int(row_index) - 1] = float(element)
		print(matrix.shape)
		print(matrix[matrix != 0.0].shape)
		self.to_csr(matrix)

	@staticmethod
	def gen_sym_positive_matrix(shape):
		a = np.random.random(shape)
		d = np.eye(shape[0]) * shape[0]
		return np.dot(np.dot(a, d), a.T)

	@staticmethod
	def get_random_right(shape, zero=True):
		if zero:
			return np.zeros(shape)
		return np.random.random(shape)

	def multiply(self, vector: np.array):
		c = np.full(self.size[0], 0.0)
		for i in range(c.shape[0]):
			if i == c.shape[0] - 1:
				low_bound, high_bound = self.LI[i], self.LI[i + 1] - 1
			else:
				low_bound, high_bound = self.LI[i], self.LI[i + 1]
			for j in range(low_bound, high_bound):
				c[i] += float(self.A[j] * vector[self.LJ[j]])
		return c.reshape(1, -1).T

	def solve(self, vector):
		x = [np.full(vector.shape, 1)]
		r = [vector - self.multiply(x[0])]
		z = [r[0]]
		k = 0
		norm_r = np.linalg.norm(r[-1], ord=None)
		norm_v = np.linalg.norm(vector, ord=None)
		norms = [norm_r]
		while norm_r / norm_v > 1e-4 and k < 1000:
			alpha_k = np.dot(r[k].T, r[k]) / (np.dot(self.multiply(z[k]).T, z[k]))
			x_k = x[k] + alpha_k * z[k]
			r_k = r[k] - alpha_k * self.multiply(z[k])
			beta_k = np.dot(r_k.T, r_k) / np.dot(r[k].T, r[k])
			z_k = r_k + beta_k * z[k]
			x.append(x_k)
			r.append(r_k)
			z.append(z_k)
			k += 1
			print(k)
			norm_r = np.linalg.norm(r[-1], ord=None)
			norms.append(norm_r)
			print(norm_r)
		# print('result x')
		# print(x[-1])
		return x, norms


if __name__ == '__main__':
	matrix = np.array([
		[0, -2, 0, 2, 0, 0],
		[0, 0, 0, 0, 0, 0],
		[0, 5, 0, 0, 0, 1],
		[0, 0, 0, 0, -9, 1],
		[0, 0, 0, 0, 0, 0],
		[0, 8, 0, 4, 0, 3]])

	print('matrix')
	print(matrix)
	solver = Solver()
	#solver.to_csr(matrix)
	#print(solver.A)
	#print(solver.LJ)
	#print(solver.LI)
	print('from file', 'bcsstk16.mtx')
	solver.csr_from_file('bcsstk16.mtx')
	#print('gen new matrix')
	#new_matrix = Solver.gen_sym_positive_matrix((1000, 1000))
	#print('gen x')
	x = np.random.random((solver.size[0], 1))
	print('gen b')
	b = solver.multiply(x)
	print('solving')
	history, norms = solver.solve(b)
	history_plot = np.array([history[0].T[0]])
	for pair in history[1:]:
		history_plot = np.vstack((history_plot, pair.T[0]))
	plt.plot(history_plot[:, 0], history_plot[:, 1])
	plt.scatter(history_plot[:, 0], history_plot[:, 1])
	plt.scatter(x[0], x[1], color='red')
	plt.grid(True)

	plt.figure()
	plt.plot(norms)
	plt.show()
