"""
Matplotlib 

Ouline
# 1. Basic
# (a) Setting style
# (b) Show the graph in
    # i. Script
    # ii. IPyhon Shell
    # iii. IPython Notebook
# (c) Save as a file
    # i. Display
    # ii. List file types supported
# (d) Bi-interface
    # i. MATLAB
    # ii. Object-oriented Interface- better when the setting of a figure is complex
# (e) Simple plot
# (f) Adjust
    # i. Line col
    # ii. Line type
    # iii. Line type + col
    # iv. Axes range
    # v. Title, Labels, Legend

# 2. Scatter Plot
# (a) plt.plot
# (b) plt.scatter
    # ex. iris dataset

# 3. Error Bar Plot

# 4. Contour plot
# (a) contour plot
# (b) image figure
# (c) Label, Overlay two

# 5. Histogram
# (a) 1D
    # i.
    # ii.
# (b) 2D
    # i. plt.hist2d
    # ii. plt.hexbin

# 6. Legend
    # ex. iris dataset
    
# 7. colorbar
    
# 8. plt.subplot

# 9. Layout- plt.GridSpec
grid = plt.GridSpec(2, 3, hspace = 0.4, wspace = 0.4)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2:])
#
mean = [0, 0]
cov = [[1, 0], 
       [2, 3]]
x, y = np.random.multivariate_normal(mean, cov, 100).T

grid = plt.GridSpec(4, 4, hspace = 0.4, wspace = 0.6)
xy = plt.subplot(grid[1:, 1:])
xy.plot(x, y, 'og')
x_dist = plt.subplot(grid[0, 1:], yticklabels = [], sharex = xy)
x_dist.hist(x, 5, histtype = 'stepfilled', 
            color = 'gray', alpha = 0.5, 
            orientation = 'vertical')

y_dist = plt.subplot(grid[1:, 0], xticklabels = [], sharey = xy)
y_dist.hist(y, 5, histtype = 'stepfilled', 
            color = 'gray', alpha = 0.5, 
            orientation = 'horizontal')
y_dist.invert_xaxis()


# 10. Arrows and Annotation

# 11. 3D plot
# (a) scatter3D
# (b) contour3D
# (c) Wire frame plot
# (d) Surface plot

"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. Basic
# (a) Setting style
plt.style.use('classic')

# (b) Show the graph in
# i. Script
import numpy as np
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
# ii. IPyhon Shell
# %matplotlib
import matplotlib.pyplot as plt
plt.plot(x, np.sin(x))
plt.draw() # forcibly update
# iii. IPython Notebook
# %matplotlib notebook　
    # Using Time: dynamic graph
%matplotlib inline 
    # static graph
fig = plt.figure()
plt.plot(x, np.sin(x), ':')
plt.plot(x, np.cos(x), '-')

# (c) Save as a file
fig.savefig('03_1 Plot.png')
# i. Display
from IPython.display import Image
Image('03_1 Plot.png')
# ii. List file types supported
fig.canvas.get_supported_filetypes() 

# (d) Bi-interface
# i. MATLAB
plt.figure()
plt.subplot(2, 1, 1) # 2 rows, 2 columns, 1st panel
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
# ii. Object-oriented Interface- better when the setting of a figure is complex
fig, ax = plt.subplots(2)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

# (e) Simple plot
%matplotlib inline
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 50)
ax.plot(x, np.sin(x))
# plt.plot(x, np.sin(x))

# (f) Adjust
# i. Line col
plt.plot(x, np.sin(x), color = 'blue') 
plt.plot(x, np.sin(x), color = 'b') # blue
plt.plot(x, np.sin(x), color = '0.5') # Gray scale, 0~1
plt.plot(x, np.sin(x), color = '#FFDD44') # RRGGBB, 00~FF
plt.plot(x, np.sin(x), color = (0.2, 1, 0.5)) # RGB tuple, 0~1
# ii. Line type
plt.plot(x, np.sin(x), linestyle = '-') # solid
plt.plot(x, np.sin(x), linestyle = '--') # dashed
plt.plot(x, np.sin(x), linestyle = ':') # dotted
plt.plot(x, np.sin(x), linestyle = '-.') # dashdot
# iii. Line type + col
plt.plot(x, np.sin(x), '-g')
plt.plot(x, np.sin(x), '-.r')

# iv. Axes range
plt.plot(x, np.sin(x))
plt.xlim(0, 2*np.pi)
plt.ylim(-1.5, 1.5);

plt.plot(x, np.sin(x))
plt.axis([0, 2*np.pi, -1.5, 1.5]);

plt.plot(x, np.sin(x))
plt.axis('tight');

plt.plot(x, np.sin(x))
plt.axis('equal'); # Figure's ratio is given output resolution

# v. Title, Labels, Legend
plt.plot(x, np.sin(x))
plt.title("Sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)");

plt.plot(x, np.sin(x), '--g', label = 'sin(x)')
plt.plot(x, np.cos(x), label = 'cos(x)')
plt.axis('equal')
plt.legend();

ax = plt.axes()
ax.plot(x, np.sin(x), '--g', label = 'sin(x)')
ax.plot(x, np.cos(x), label = 'cos(x)')
ax.set(xlim = (0, 2*np.pi), ylim = (-1.5, 1.5),
       xlabel = 'x', ylabel = '', title = 'Sin(x), Cos(x)')
ax.legend();


# 2. Scatter Plot
# (a) plt.plot
%matplotlib inline
plt.style.use('seaborn-whitegrid')
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x), 'o', color = "k")
plt.plot(x, np.sin(x), '-og') # lty, pt type, col
plt.plot(x, np.sin(x), '-sg', 
         linewidth = 5, 
         markersize = 10, markerfacecolor = 'w',
         markeredgewidth = 2, markeredgecolor = 'purple')

# (b) plt.scatter
# ex. iris dataset
from sklearn.datasets import load_iris
iris_load = load_iris()
var_name = iris_load.feature_names
iris_data = iris_load.data
iris_class = iris_load.target
import pandas as pd
iris = pd.DataFrame(iris_data, columns = var_name)
iris.head()
plt.scatter(iris[var_name[0]], iris[var_name[1]],
            c = iris_class, cmap = 'viridis', alpha = 0.4,
            s = iris[var_name[3]]*100)
plt.xlabel(var_name[0])
plt.ylabel(var_name[1])
plt.title('Iris')
# plt.colorbar() # useful when colors of pts are given numerical attr. values


# 3. Error Bar Plot
%matplotlib inline
plt.style.use('seaborn-whitegrid')
x = np.arange(15) + 1
y_bar = np.random.normal(0, 1/15, 15)
std_error = 1.96 * 1/np.sqrt(15) # 95% C.I.
plt.errorbar(x, y_bar, yerr = std_error, fmt = 'db',
             ecolor = 'g', elinewidth = 2, capsize = 5)


# 4. Contour plot
# (a) contour plot
%matplotlib inline
plt.style.use('seaborn-whitegrid')
x = y = np.linspace(-10, 10)
X, Y = np.meshgrid(x, y)
Z = X**3 + Y**3
plt.contour(X, Y, Z, colors = "k") # solid: positive num, dashed: negative num
plt.contour(X, Y, Z, 20, colors = "k") # 20: more lines shown
plt.contour(X, Y, Z, 20, cmap = 'RdGy') # cmap: col mapping # RdGy: Red-gray scale
plt.contourf(X, Y, Z, 20, cmap = 'RdGy') # 'F'ill up with colors
plt.colorbar(label = 'Z values')
# plt.colorbar().set_label('Z values')

# (b) image figure
plt.imshow(Z, extent = [-10, 10, -10, 10], 
           origin = 'lower',  # Adjust the loc. of origin (topleft in default)
           cmap = 'RdGy')
plt.colorbar()
plt.axis(aspect = 'image') # Figure ratio caters to units of x and y

# (c) Label, Overlay two
contours = plt.contour(X, Y, Z, 20, colors = "k")
plt.clabel(contours, inline = True, fontsize = 8)
plt.imshow(Z, extent = [-10, 10, -10, 10], 
           origin = 'lower', cmap = 'RdGy', alpha = 0.5)
plt.colorbar()


# 5. Histogram
# (a) 1D
# i.
%matplotlib inline
plt.style.use('seaborn-whitegrid')
data = np.random.randn(100)
plt.hist(data, bins = 30)
count, bin_edges = np.histogram(data , bins = 30)
count
plt.hist(data, bins = 30, normed = True, # normed = T: prob density
         color = 'g', alpha = 0.5, edgecolor = 'none')
count/count.sum()
# ii.
x1 = np.random.normal(0, 0.8, 100)
x2 = np.random.normal(-4, 1, 100)
x3 = np.random.normal(3, 5, 100)
kwargs = dict(bins = 10, normed = True, color = 'g', alpha = 0.3, 
              histtype = 'stepfilled')
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

# (b) 2D
mean = [0, 0]
cov = [[1, 0], 
       [2, 3]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
# i. plt.hist2d
plt.hist2d(x, y, bins = 30, cmap = 'Blues')
plt.colorbar(label = 'counts')

# ii. plt.hexbin
plt.hexbin(x, y, gridsize = 30, cmap = 'Blues')
plt.colorbar(label = 'counts')


# 6. Legend
%matplotlib inline
plt.style.use('classic')
x = np.linspace(-10, 10, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-k', label = 'sin(x)')
ax.plot(x, np.cos(x), ':g', label = 'cos(x)')
ax.axis('equal')
# ax.legend()
# ax.legend(loc = 'upper left', frameon = False, ncol = 2)
ax.legend(fancybox = True, framealpha = 0.3, shadow = True, borderpad = 2)

# ex. iris dataset
"""Recall # 2.(b) --------------------------------
from sklearn.datasets import load_iris
iris_load = load_iris()
var_name = iris_load.feature_names
iris_data = iris_load.data
iris_class = iris_load.target
import pandas as pd
iris = pd.DataFrame(iris_data, columns = var_name)
iris.head()
---------------------------------------"""
%matplotlib inline
plt.style.use('seaborn-whitegrid')
plt.scatter(iris[var_name[0]], iris[var_name[1]],
            c = iris_class, cmap = 'viridis', alpha = 0.4,
            s = iris[var_name[3]]*100, label = None)
plt.xlabel(var_name[0])
plt.ylabel(var_name[1])
plt.title('Iris')
for i in [0.5, 1, 1.5]:
    plt.scatter([], [], c = 'k', alpha = 0.3, s = i*100,
                label = str(i) + ' cm')
plt.legend(scatterpoints = 1, labelspacing = 0.2, 
           title = var_name[3], ncol = 3)


# 7. colorbar
"""Reacall # 4.(b) ----------------------------
plt.imshow(Z, extent = [-10, 10, -10, 10], 
           origin = 'lower',  cmap = 'RdGy')
plt.colorbar()
-------------------------------------------"""
plt.imshow(Z, extent = [-10, 10, -10, 10], 
           origin = 'lower',  cmap = 'RdBu')
plt.colorbar(extend = 'both')

plt.imshow(Z, extent = [-10, 10, -10, 10], 
           origin = 'lower',  cmap = plt.cm.get_cmap('RdBu', 9))
plt.colorbar(extend = 'both')


# 8. plt.subplot
%matplotlib inline
plt.style.use('seaborn-white')
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2]) # [bottom, left, width(%), height(%)]
# Oriented object inference
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels = [], ylim = (-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim = (-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
# i.
plt.subplots_adjust(hspace = 0.4, wspace = 0.4) # adjust margins(%)
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((i)),
             fontsize = 18, ha = 'center') # ha: horizontal alignment
# ii.
fig, ax = plt.subplots(2, 3, sharex = 'col', sharey = 'row')
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), 
                      fontsize = 18, ha = 'center')


# 9. Layout- plt.GridSpec
grid = plt.GridSpec(2, 3, hspace = 0.4, wspace = 0.4)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2:])
#
mean = [0, 0]
cov = [[1, 0], 
       [2, 3]]
x, y = np.random.multivariate_normal(mean, cov, 100).T

grid = plt.GridSpec(4, 4, hspace = 0.4, wspace = 0.6)
xy = plt.subplot(grid[1:, 1:])
xy.plot(x, y, 'og')
x_dist = plt.subplot(grid[0, 1:], yticklabels = [], sharex = xy)
x_dist.hist(x, 5, histtype = 'stepfilled', 
            color = 'gray', alpha = 0.5, 
            orientation = 'vertical')

y_dist = plt.subplot(grid[1:, 0], xticklabels = [], sharey = xy)
y_dist.hist(y, 5, histtype = 'stepfilled', 
            color = 'gray', alpha = 0.5, 
            orientation = 'horizontal')
y_dist.invert_xaxis()


# 10. Arrows and Annotation
x = np.arange(10)
y = np.zeros(10)
plt.plot(x, y, '-og', linewidth = 2)
plt.xlim(-0.5, 9.5)
"""
xy: arrow's head
xytext: arrow's tail
arrowprops: arrow type
    fc: face color
    ec: edge color
    shrink: shrinkage rate of length   
    arrowstyle: arrow style
    (*-- https://matplotlib.org/users/annotations_guide.html--*)
        ->, -|>, <-, <|-, <->, <|-|>,
        -[, ]-, ]-[, |-|, fancy, simple, wedge
    connectionstyle: connection style
    (*-- https://matplotlib.org/users/annotations_guide.html--*)
        angle, angle3, arc, arc3, bar 
bbox
    boxstyle: box style
        circle, darrow, larrow, rarrow, 
        square, round, round4, roundtooth, sawtooth,        
    fc
    ec  
size: font size and box size

"""
# i.
plt.annotate('1st', xy = (0, 0), xytext = (0, -0.02),
             arrowprops = dict(fc = 'g', ec = 'b', alpha = 0.5,
                               shrink = 0.1))
# ii.
plt.annotate('2nd', xy = (1, 0), xytext = (2, 0.04),
             arrowprops = dict(arrowstyle = '->',
                               connectionstyle = 'angle,angleA=0,angleB=90'))
# iii.
plt.annotate('3rd', xy = (2, 0), xytext = (1, -0.04),
             arrowprops = dict(arrowstyle = 'fancy', fc = 'k'))
# iv.
plt.annotate('4th', xy = (3, 0), xytext = (2, -0.05),
             arrowprops = dict(arrowstyle = '<-',
                               connectionstyle = 'angle3'))
# v.
plt.annotate('5th', xy = (4, 0), xytext = (4, 0.02), size = 13, 
             arrowprops = dict(arrowstyle = 'wedge, tail_width = 0.5', 
                               fc = 'g', alpha = 0.2),
             bbox = dict(boxstyle = 'round4', fc = 'g', alpha = 0.2))


# 11. 3D plot
from mpl_toolkits import mplot3d
%matplotlib inline
fig = plt.figure()
ax = plt.axes(projection = '3d')

# (a) scatter3D
zl = np.linspace(0, 15, 1000)
xl = np.sin(zl)
yl = np.cos(zl)
ax.plot3D(xl, yl, zl, 'gray')

z = 15*np.random.random(100)
x = np.sin(z) + 0.1 * np.random.randn(100)
y = np.cos(z) + 0.1 * np.random.randn(100)
ax.scatter3D(x, y, z, c = z, cmap = 'Reds')

# (b) contour3D
# v.s 4.
x = y = np.linspace(-10, 10, 30)
X, Y = np.meshgrid(x, y)
# i.
Z = X**3 + Y**3
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z, 50, cmap = 'Blues')
# ii.
Z2 = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.contour3D(X, Y, Z2, 50, cmap = 'Blues')
ax.view_init(60, 35) # view

# (c) Wire frame plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X, Y, Z2, color = 'g', alpha = 0.3)

# (d) Surface plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z2, cmap = 'Blues', 
                edgecolor = 'none',
                rstride = 1, cstride = 1)





