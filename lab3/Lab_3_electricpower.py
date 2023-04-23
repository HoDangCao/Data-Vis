# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 02:33:19 2015

@author: nymph
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


############################## Your code for loading and preprocess the data ##
df = pd.read_csv('exdata_data_household_power_consumption.zip', delimiter=';')
df = df[(df['Date'] == '1/2/2007') + (df['Date'] == '2/2/2007')].reset_index(drop=True)
df.iloc[:,2:] = df.iloc[:,2:].astype('float64')
# df.Date = pd.to_datetime(df.Date)

############################ Complete the following 4 functions ###############
def plot1():
    plt.hist(df.Global_active_power, bins=np.arange(0, int(df.Global_active_power.max())+0.1, 0.5), edgecolor='black')
    plt.title("Global Active Power")
    plt.xlabel("Global Active Power (kilowatts)")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.savefig('plot1.png')
    # plt.show()

def line_plot(plt, attribute):
    time = pd.to_datetime(df.Date + ' ' + df.Time, dayfirst=True)
    hour = time.apply(lambda t: (t - time[0]).total_seconds()/3600)

    if type(attribute) != list:
        plt.plot(hour, df[attribute])
        plt.ylabel(attribute)
    else:
        for attr in attribute:
            plt.plot(hour, df[attr], label=attr)
        plt.ylabel("Energy Sub Metering")
        plt.legend()

    plt.xlabel(f"Hour (from {time[0]})")
    plt.xticks(hour[::60])
    return plt

def plot2():
    plt.figure().set_figwidth(15)
    plot = line_plot(plt, 'Global_active_power')
    plot.savefig('plot2.png')
    # plot.show()

def plot3():
    plt.figure().set_figwidth(15)
    plot = line_plot(plt, [f"Sub_metering_{i+1}" for i in range(3)])
    plot.savefig('plot3.png')
    # plot.show()

def plot4():
    plt.figure(figsize=(30,10))
    plt.tight_layout()
    
    plt.subplot(2,2,1)
    line_plot(plt, 'Global_active_power')
    plt.subplot(2,2,2)
    line_plot(plt, 'Voltage')
    plt.subplot(2,2,3)
    line_plot(plt, [f"Sub_metering_{i+1}" for i in range(3)])
    plt.subplot(2,2,4)
    plot = line_plot(plt, 'Global_reactive_power')

    plot.savefig('plot4.png')
    # plot.show()

plot1()
plot2()
plot3()
plot4()