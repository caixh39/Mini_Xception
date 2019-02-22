import re
import os
import sys
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.random import beta
from pandas import DataFrame, Series
from chart_func import single_people_number

# data: imitate 4 emotions, record neutral,happy, sad, angry, fear;
hudi = [[30, 31, 18, 44], [81, 4, 18, 0], [0, 75, 38, 23], [2, 0, 48, 2], [0, 0, 0, 44]]
caixh = [[35, 69, 16, 35], [75, 1, 2, 10], [0, 20, 76, 58], [0, 20, 8, 3], [0, 1, 11, 3]]
wangyue = [[24, 34, 62, 50], [93, 0, 6, 0], [0, 26, 36, 43], [0, 53, 10, 7], [0, 0, 0, 7]]
jiangzijian = [[51, 94, 48, 0], [56, 5, 11, 0], [0, 2, 3, 3, 0], [2, 2, 38, 47], [0, 0, 3, 0]]
sunjunwen = [[42, 91, 40, 61], [59, 2, 0, 7], [0, 19, 60, 39], [1, 0, 12, 0], [0, 0, 0, 0]]


def plot_beta_hist(ax, a, b):
    ax.hist(beta(a, b, size=10000), histtype="stepfilled",
            bins=25, alpha=0.8, normed=True)


def sigle_proability():
    plt.style.use('bmh')

    fig, ax = plt.subplots()
    plot_beta_hist(ax, 10, 10)
    plot_beta_hist(ax, 4, 12)
    plot_beta_hist(ax, 50, 12)
    plot_beta_hist(ax, 6, 55)
    ax.set_title("'bmh' style sheet")

    plt.show()


if __name__ == '__main__':
    # single one person's info
    # single_people_number(user=hudi, n_groups=4)
    file_path = './'
    csv_file = file_path + 'hudi_1.csv'
    emotions_result = pd.read_csv(csv_file)


    for i in range(0, len(emotions_result.Time)):
        row = emotions_result.iloc[i]['Time']
        # print row
        a = int(row)

        time = datetime.utcfromtimestamp(a)
        time = time.strftime("%Y-%m-%d %H:%M:%S")


    happy = []
    sad = []
    angry = []
    fear = []
    neutral = []

    happy_p = []
    sad_p = []
    angry_p = []
    fear_p = []
    neutral_p = []

        # print(time)
    for i in range(len(emotions_result.Emotion)):
        if emotions_result.Emotion[i] == 'happy':
            print emotions_result.Emotion[i]
            print emotions_result.Probability[i]
            happy.append(emotions_result.Emotion[i])
            happy_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'sad':
            print emotions_result.Emotion[i]
            print emotions_result.Probability[i]
            sad.append(emotions_result.Emotion[i])
            sad_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'angry':
            print emotions_result.Emotion[i]
            print emotions_result.Probability[i]
            angry.append(emotions_result.Emotion[i])
            angry_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'fear':
            print emotions_result.Emotion[i]
            print emotions_result.Probability[i]
            fear.append(emotions_result.Emotion[i])
            fear_p.append(emotions_result.Probability[i])
        else:
            print emotions_result.Emotion[i]
            print emotions_result.Probability[i]
            neutral.append(emotions_result.Emotion[i])
            neutral_p.append(emotions_result.Probability[i])


            # os._exit(0)




    # differents emotion values
    # label_happy = emotions_result.Happy
    # proability_happy = emotions_result.Proability_h

    # # print emotions_result.Proability_h

    # label_sad = emotions_result.Sad
    # proability_sad = emotions_result.Proability_s

    # label_angry = emotions_result.Angry
    # proability_angry = emotions_result.Proability_a

    # label_fear = emotions_result.Fear
    # proability_fear = emotions_result.Proability_f

    # sigle_probability()
    # data = []
    # # print emotions_result.Time
    # time = emotions_result.Time
    # for i in range(len(time)):
    #     data[i] = datetime.strptime(time[i], '%Y-%m-%d')
    # print data['time']
    # data.set_index('time')

    # data['Proability'].plot()

