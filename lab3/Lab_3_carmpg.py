# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 21:53:34 2015

@author: nymph
"""


#################################### Read the data ############################
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import numpy as np

''' read_csv()
The read_csv() function in pandas package parse an csv data as a DataFrame data structure. What's the endpoint of the data?
The data structure is able to deal with complex table data whose attributes are of all data types. 
Row names, column names in the dataframe can be used to index data.
'''

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original", delim_whitespace = True, \
 header=None, names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin', 'car_name'])

data['mpg']
data.mpg
data.iloc[0,:]

print(data.shape)

################################## Enter your code below ######################
print(data)

# 1. How many cars and how many attributes are in the data set.
print(f"1. There are {data.shape[0]} cars and {data.shape[1]} attributes.")


# 2. How many distinct car companies are represented in the data set? 
data['company'] = list(zip(*data['car_name'].str.split()))[0]
print(f"2. There are {data['company'].drop_duplicates().count()} distinct car companies are represented in the data set")

# What is the name of the car with the best MPG? 
print(f"- {data[data.mpg == data.mpg.max()].car_name.values[0]} is the car with the best MPG.")

# What car company produced the most 8-cylinder cars? 
num_8_cyl_com = data[data.cylinders == 8.0][['company','car_name']].groupby(['company']).count()
print(f"- {num_8_cyl_com[num_8_cyl_com.car_name == num_8_cyl_com.car_name.max()].index[0]} produced the most 8-cylinder cars.")

# What are the names of 3-cylinder cars? 
print(f"- The names of 3-cylinder cars: {list(data[data.cylinders == 3].car_name)}")

# Do some internet search that can tell you about the history and popularity of those 3-cylinder cars.



# 3. What is the range, mean, and standard deviation of each attribute? Pay attention to potential missing values.
print("3. Description of numeric attribute and potential missing values")
miss_rate = Series(data.isnull().mean() * 100, name='missing_rate')
print(data.describe().append(miss_rate))


import matplotlib.pyplot as plt
# 4. Plot histograms for each attribute. Pay attention to the appropriate choice of number of bins.
def plot_hist(attribute):
    plt.hist(data[attribute], edgecolor = "black")
    plt.title(f'Distribution of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('quantity')
    plt.grid(axis='y')
    plt.show()

print('4.')
for col in data.columns[:-2]:
    plot_hist(col)

# Write 2-3 sentences summarizing some interesting aspects of the data by looking at the histograms.



# 5. Plot a scatterplot of weight vs. MPG attributes.
# What do you conclude about the relationship between the attributes? 
# What is the correlation coefficient between the 2 attributes?
def plot_scatter(attr_1, attr_2, df):
    # comment
    value = df[attr_1].corr(df[attr_2])
    type_corr, go = ("negative", "decrease") if value < 0 else ("positive","increase")
    print(f'When we look at the entire chart, we see that {attr_1} rises, {attr_2} tends to {go}.')
    print(f'The correlation coefficient between {attr_1} and {attr_2} is {type_corr} and equal: {value}\n')

    plt.scatter(df[attr_1], df[attr_2] + np.random.random(len(df[attr_2])))
    plt.title(f'Scatter plot of {attr_1} and {attr_2}')
    plt.xlabel(attr_1)
    plt.ylabel(attr_2)
    plt.show()

print('5.')
plot_scatter('weight', 'mpg', data)

# 6. Plot a scatterplot of year vs. cylinders attributes. 
# Add a small random noise to the values to make the scatterplot look nicer. 
# (Hint: data.mpg + np.random.random(len(data.mpg)) will add small random noise)
# What can you conclude?
print('6.')
plot_scatter('model', 'cylinders', data)

print("-----Search about the history of car industry during 70’s-----")
print("Source about data https://www.retrowaste.com/1970s/cars-in-the-1970s/")
data6 = data[data['model'] < 80]
plot_scatter('model', 'acceleration',data6)
plot_scatter('model', 'displacement',data6)
plot_scatter('model', 'horsepower',data6)
plot_scatter('model', 'weight', data6)

# Do some internet search about the history of car industry during 70’s that might explain the results.
# Source about data https://www.retrowaste.com/1970s/cars-in-the-1970s/
print("--> The evidence on the internet has partly explained the change of properties in the data that we have visualized. The things that we realize after visualizing completely match the evidence on the internet")


# 7. Show 2 more scatterplots that are interesting do you. Discuss what you see. (discuss in report)
plot_scatter('cylinders', 'displacement', data)
plot_scatter('acceleration', 'horsepower', data)


# 8. Plot a time series for all the companies that show how many new cars they introduces during each year.
# Do you see some interesting trends? (Hint: data.car name.str.split()[0] returns a vector of the first word of car name column.)
print('8.')
df = data[['model','company']].groupby(['model','company']).value_counts().unstack(level=-1)
df = df.fillna(0)
df.plot(figsize=(12,10))
plt.title("Number of new cars is introduced during each year")
plt.xticks(df.index)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.legend(bbox_to_anchor=(1.18, 1), loc=0)
plt.show()


# 9. Calculate the pairwise correlation, and draw the heatmap with Matplotlib. 
# Do you see some interesting correlation? (Hint: data.iloc[:,0:8].corr(), plt.pcolor() draws the heatmap.)
print('9.')
print(
"""
- cylinders, displacment, horsepower and weight have high positive correlation each other.
- cylinders, displacment, horsepower and weight have high negative correlation mpg.
""")
corr = data.iloc[:,0:8].corr()
heat_map = plt.pcolor(corr.corr())
plt.xticks(np.arange(0.5, len(corr.index)), labels=corr.index, rotation = 45)
plt.yticks(np.arange(0.5, len(corr.index)), labels=corr.index)
plt.colorbar(heat_map)
plt.show()