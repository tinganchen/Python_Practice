"""
Seaborn 

Ouline
"""
import seaborn as sns
sns.set()

# 1. Histogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 1], [2, 1]]
data = np.random.multivariate_normal(mean, cov, 1000)
data = pd.DataFrame(data, columns = ['x', 'y'])
# (a) 1D
# i. hist
for col in 'xy':
    plt.hist(data[col], normed = True, 
             color = 'g', alpha = 0.4)
# ii. kde
for col in 'xy':
    sns.kdeplot(data[col], shade = True)
# iii. distplot
for col in 'xy':
    sns.distplot(data[col])

# (b) 2D
# i. kde
sns.kdeplot(data)
# ii. joint_kde
with sns.axes_style('white'):
    sns.jointplot('x', 'y', data, kind = 'kde')
# iii. joint_hex
sns.jointplot('x', 'y', data, kind = 'hex')


# 2. Pair plot
# e.x. iris dataset
iris = sns.load_dataset('iris')
iris.head()
sns.pairplot(iris, hue = 'species', size = 2.5)

# 3. EX. tips dataset
tips = sns.load_dataset('tips')
tips.head()
tips['tips_pct'] = tips['tip'] / tips['total_bill'] * 100
# (a) Faceted histogram
grid = sns.FacetGrid(tips, row = 'sex', col = 'time',
                     margin_titles = True)
grid.map(plt.hist, 'tips_pct', bins = np.linspace(0, 40, 10))

# (b) Factor plot, Boxplot
box = sns.factorplot('day', 'total_bill', 'sex', 
                     data = tips, kind = 'box')
box.set_axis_labels('Day', "Total Bill")

# (c) Joint plot
# i. hex
sns.jointplot('total_bill', 'tip', data = tips, kind = 'hex')
# ii. Regression
sns.jointplot('total_bill', 'tip', data = tips, kind = 'reg')














