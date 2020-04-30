import numpy as np

array = np.ones((9,13))
array = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4],
                 [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 4, 4], [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 4, 4], [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 4, 4],
                 [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 4, 4], [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 4, 4], [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 4, 4]])
print("SHAPE 1:", array.shape)
filter_size = 3
y_times = int(array.shape[0] / filter_size)
x_times = int(array.shape[1] / filter_size)
print("y: ", y_times)
print("x: ", x_times)
array_filtered = np.zeros((y_times, x_times))
for y in range(y_times):
    index_y = y*filter_size
    for x in range(x_times):
        index_x = x*filter_size
        array_filtered[y, x] = np.mean(array[index_y:index_y+filter_size, index_x:index_x+filter_size])
        print("array_filtered[%s, %s] = np.mean(array[%s:%s, %s:%s]" % (y, x, index_y, index_y+filter_size, index_x, index_x+filter_size))

print(array_filtered)