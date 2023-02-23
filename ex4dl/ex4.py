import numpy as np
import matplotlib.pyplot as plt

def cls_node(data, t, map, m_rows, m_cols):
  res = (0,0)
  s_dis = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(map[i][j], data[t])
      if ed < s_dis:
        s_dis = ed
        res = (i, j)
  return res

def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2) 

def man_dis(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)

def mst_cmn(lst, n):
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)

def main():
  np.random.seed(1)
  Dim = 4
  Rows = 30; Cols = 30
  RangeMax = Rows + Cols
  LearnMax = 0.5
  StepsMax = 5000

  print("\nLoading Iris data into memory \n")
  data_file = "iris.txt"
  data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,4),
    dtype=np.float64)
  data_y = np.loadtxt(data_file, delimiter=",", usecols=[4],
    dtype=np.int)

  print("Constructing a 30x30 SOM from the iris data")
  map = np.random.random_sample(size=(Rows,Cols,Dim))
  for s in range(StepsMax):
    if s % (StepsMax/10) == 0: print("step = ", str(s))
    pct_left = 1.0 - ((s * 1.0) / StepsMax)
    curr_range = (int)(pct_left * RangeMax)
    curr_rate = pct_left * LearnMax

    t = np.random.randint(len(data_x))
    (bmu_row, bmu_col) = cls_node(data_x, t, map, Rows, Cols)
    for i in range(Rows):
      for j in range(Cols):
        if man_dis(bmu_row, bmu_col, i, j) < curr_range:
          map[i][j] = map[i][j] + curr_rate * \
(data_x[t] - map[i][j])
  print("SOM construction complete \n")

  print("Constructing U-Matrix from SOM")
  u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
  for i in range(Rows):
    for j in range(Cols):
      v = map[i][j]
      sum_dists = 0.0; ct = 0
     
      if i-1 >= 0:
        sum_dists += euc_dist(v, map[i-1][j]); ct += 1
      if i+1 <= Rows-1:
        sum_dists += euc_dist(v, map[i+1][j]); ct += 1
      if j-1 >= 0:
        sum_dists += euc_dist(v, map[i][j-1]); ct += 1
      if j+1 <= Cols-1:
        sum_dists += euc_dist(v, map[i][j+1]); ct += 1
      
      u_matrix[i][j] = sum_dists / ct
  print("U-Matrix constructed \n")

  plt.imshow(u_matrix, cmap='gray')
  plt.show()

  print("Associating each data label to one map node ")
  mapping = np.empty(shape=(Rows,Cols), dtype=object)
  for i in range(Rows):
    for j in range(Cols):
      mapping[i][j] = []

  for t in range(len(data_x)):
    (m_row, m_col) = cls_node(data_x, t, map, Rows, Cols)
    mapping[m_row][m_col].append(data_y[t])

  label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
  for i in range(Rows):
    for j in range(Cols):
      label_map[i][j] = mst_cmn(mapping[i][j], 3)
 
  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4))
  plt.colorbar()
  plt.show()


if __name__=="__main__":
  main()
