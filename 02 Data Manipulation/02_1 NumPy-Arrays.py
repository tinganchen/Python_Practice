"""
NumPy, Numerical Python

Outline

# 1. Create Arrays- array()
# (a) One-dimensional Array
# (b) 2D Array
# (c) Define data type of element in arrays
# (d) reshape()
# (e) Other functions
# arange()
# zeros()
# zeros_like()
# ones()
# ones_like()
# full()
# linspace()-seq
# eye()- identity matrix and its variants
# random.rand()
# empty()

# 2. Properties- dtype, size, shape, ndim, itemsize, nbytes, astype

# 3. I/O
# (a) Save and Load on disk
# (b) Save and Load text files

# 4. Array Mathematics
# (a) Arithmetic Operations
## 1D
# array vs. scalar
# array vs. array
## 2D
# array vs. scalar
## Scalar

# 5. Comparison
# (a) Element-wise
# (b) Array-wise

# 6. Aggregate Functions

# 7. Copy, Fill, Concatenate Arrays

# 8. Sort Arrays

# 9. Subsetting, Slicing, Indexing
# (a) Subset
# (b) Slice
# (c) Index
# Boolean Indexing
# Fancy Indexing

# 10. Broadcasting

# 11. Other Array Manipulation
# (a) transpose
# (b) Changing Array Shape
# (c) np.random
# (d) Adding/Removing Elements
# (e) Combining Arrays
# (f)  Splitting Arrays

"""
import numpy as np

# 1. Create Arrays- array()
# (a) One-dimensional Array
a = np.array([1, 2, 3])
print(a)
type(a)
print(a[1])
a[1] = 0
print(a)

# (b) 2D Array
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
type(b)
print(b[0, 2])
b[0, 0] = 0
print(b)

# (c) Define data type of element in arrays
c = np.array([1, 2, 3], dtype = float) # int

# (d) reshape()
np.arange(6).reshape((2, 3))

# (e) Other functions
# arange()
np.arange(5) # array range
np.arange(1, 9, 3)

# zeros()
np.zeros(3)
np.zeros((2, 3))
# zeros_like()
np.zeros_like(b)

# ones()
np.ones(3)
np.ones((2, 3))
# ones_like()
np.ones_like(b)

# full()
np.full(2, 2)
np.full((2, 3), 2)

# linspace()-seq
np.linspace(0, 1, 5)

# eye()- identity matrix and its variants
np.eye(3)
np.eye(3, k = 0) # 1, 2

# random.rand()
np.random.rand(3)
np.random.rand(2, 3)

# empty()
np.empty((3,2))


# 2. Properties- dtype, size, shape, ndim, itemsize, nbytes, astype
a = np.arange(6).reshape((2, 3))
a.dtype # nbytes = size * itemsize
a.dtype.name
a.astype(float)


# 3. I/O
# (a) Save and Load on disk
a = np.arange(6).reshape((2, 3))
b = np.arange(6)
np.save('02_1 my_array', a) # .npy file in default
np.load('02_1 my_array.npy')

np.savez('02_1 arrays', a, b)
np.load('02_1 arrays.npz')

# (b) Save and Load text files
np.savetxt("02_1 myarray.txt", a, delimiter=" ")
np.loadtxt("02_1 myarray.txt")


# 4. Array Mathematics
# (a) Arithmetic Operations
## 1D
# array vs. scalar
a = np.array([1, 2, 3])
s = 2
a + s # +, -, *, /
np.add(a, s) # add, subtract, multiply, divide
# array vs. array
b = np.array([2, 2, 2])
a + b # +, -, *, /
np.add(a, b) # add, subtract, multiply, divide
a.dot(b) # dot product

## 2D
# array vs. scalar
a = np.array(([1, 2, 3], [4, 5, 6]))
s = 2
a + s # +, -, *, /
np.add(a, s) # add, subtract, multiply, divide
# array vs. array
b = np.array(([1, 1, 1], [2, 2, 2]))
a + b # +, -, *, /
np.add(a, b) # add, subtract, multiply, divide
c = np.array(([1, 1], [2, 2], [3, 3]))
a.dot(c) # dot product

## Scalar
d = 30
np.exp(d)
np.sqrt(d)
np.sin(d)
np.cos(d)
np.log(d)
np.pi
e = 1.05
np.around(e, decimals = 0)
np.floor(e)
np.ceil(e)


# 5. Comparison
# (a) Element-wise
a = np.array(([1, 2, 3], [4, 5, 6]))
b = np.array(([1, 1, 1], [2, 2, 2]))
a == b
a < b
a < 2
# (b) Array-wise
np.array_equal(a, b)

# 6. Aggregate Functions
a = np.array(([1, 2, 3], [4, 5, 6]))
b = np.array(([1, 1, 1], [2, 2, 2]))
a.sum()
a.min()
b.max(axis=0) # axis=0: column, 1: row
b.cumsum(axis=0)
a.mean()
np.median(b)
np.corrcoef(a)
np.std(b)  

c = np.arange(3)
c.sum()
np.add.reduce(c)
np.multiply.reduce(c)

np.add.outer(c, c)
np.multiply.outer(c, c)


# 7. Copy, Fill, Concatenate Arrays
a = np.array(([1, 2, 3], [4, 5, 6]))
b = np.array(([1, 1, 1], [2, 2, 2]))
a.view()
a.copy()
np.copy(a)
a.fill(0)
a
np.concatenate((a, b))

# 8. Sort Arrays
a = np.array(([3, 2, 1], [1, 3, 5]))
np.sort(a)
np.sort(a, axis=0)


# 9. Subsetting, Slicing, Indexing
# (a) Subset
a = np.array([1, 2, 3])
b = np.array(([1, 2, 3], [4, 5, 6]))
a[0]
b[1, 2]

# (b) Slice
# 1D- array[start:end:step]
# 2D- array[start:end:step, start1:end1:step1]
a = np.array([1, 2, 3])
b = np.array(([1, 2, 3], [4, 5, 6]))
a[0:2]
a[ : : -1] # rev = True
b[0:, 1] # 2nd column
b[:2, :1] # first 2 rows, first 1 col.
b[0, ...] 

# (c) Index
a = np.array([1, 2, 3])
b = np.array(([1, 2, 3], [4, 5, 6]))
# Boolean Indexing
a[a < 3]
b[b < 5]
# Fancy Indexing
a[[0, 2]]
a[range(1)]
a[[0, 1]] = 5
a
b[[1, 0, 1, 0],[0, 1, 2, 0]]
    # Select elements (1,0),(0,1),(1,2) and (0,0)

# 10. Broadcasting
a = np.arange(9).reshape((3, 3))
b = np.array([1, 2, 3])
a + b

# 11. Other Array Manipulation
# (a) transpose
a = np.arange(9).reshape((3, 3))
np.transpose(a)

# (b) Changing Array Shape
a = np.arange(9)
a.reshape((3, 3))
a.ravel()

# (c) np.random
# seed(int)
np.random.seed(123)
# random()
v1 = np.random.random()
v2 = np.random.random()
# randint(min, max, size)
v3 = np.random.randint(5, 10)
# rand(row, col)
v4 = np.random.rand(2)
v5 = np.random.rand(2, 3)
# randn(row, col)- random number from normal distribution
v6 = np.random.randn(2, 3)
# ~ Normal(0, 1)
np.random.normal(0, 1, 5)
np.random.normal(0, 1, (3, 3))

# (d) Adding/Removing Elements
a = np.arange(9)
a.resize((2, 3))
a
b = np.array(([1, 2, 3], [4, 5, 6]))
np.append(a, b)

c = np.arange(9)
np.insert(c, 1, 5)
np.delete(c,[1])

# (e) Combining Arrays
a = np.arange(9)
a.resize((2, 3))
a
b = np.array(([1, 2, 3], [4, 5, 6]))
# rbind
np.vstack((a,b))
np.row_stack((a, b))
np.r_[a, b]

# cbind
np.hstack((a,b))
np.column_stack((a, b))
np.c_[a, b]

# (f)  Splitting Arrays
a = np.arange(9)
np.hsplit(a,3)
b = np.array(([1, 2, 3], [4, 5, 6]))
np.vsplit(b, 1)