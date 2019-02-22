import re
import os
import sys
import csv
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from numpy.random import beta
from pandas import DataFrame, Series
from chart_func import single_people_number,get_single_emotion,get_single_probability

# data: imitate 4 emotions, record neutral,happy, sad, angry, fear;
hudi = [[30, 31, 18, 44], [81, 4, 18, 0], [0, 75, 38, 23], [2, 0, 48, 2], [0, 0, 0, 44]]
caixh = [[35, 69, 16, 35], [75, 1, 2, 10], [0, 20, 76, 58], [0, 20, 8, 3], [0, 1, 11, 3]]
wangyue = [[24, 34, 62, 50], [93, 0, 6, 0], [0, 26, 36, 43], [0, 53, 10, 7], [0, 0, 0, 7]]
jiangzijian = [[51, 94, 48, 0], [56, 5, 11, 0], [0, 2, 3, 3, 0], [2, 2, 38, 47], [0, 0, 3, 0]]
sunjunwen = [[42, 91, 40, 61], [59, 2, 0, 7], [0, 19, 60, 39], [1, 0, 12, 0], [0, 0, 0, 0]]


def plot_beta_hist(ax, a, b):
    ax.hist(beta(a, b, size=10000), histtype="stepfilled",
            bins=25, alpha=0.8, normed=True)


# def sigle_proability():
#     plt.style.use('bmh')

#     fig, ax = plt.subplots()
#     plot_beta_hist(ax, 10, 10)
#     plot_beta_hist(ax, 4, 12)
#     plot_beta_hist(ax, 50, 12)
#     plot_beta_hist(ax, 6, 55)
#     ax.set_title("'bmh' style sheet")

#     plt.show()

def sigle_proability(emotions_result):
    plt.style.use('bmh')
    
    for i in range(len(emotions_result.id)):
        if emotions_result.Emotion[i] == 'happy':
            # print emotions_result.Emotion[i]
            happy_p.append(emotions_result.Emotion[i])
            sad_p.append(0)
            angry_p.append(0)
            fear_p.append(0)
            neutral_p.append(0)

        elif emotions_result.Emotion[i] == 'sad':
            # print emotions_result.Emotion[i]
            happy_p.append(0)
            sad_p.append(emotions_result.Emotion[i])
            angry_p.append(0)
            fear_p.append(0)
            neutral_p.append(0)

        elif emotions_result.Emotion[i] == 'angry' or emotions_result.Emotion[i] == 'disgust':
            # print emotions_result.Emotion[i]
            happy_p.append(0)
            sad_p.append(0)
            angry_p.append(emotions_result.Emotion[i])
            fear_p.append(0)
            neutral_p.append(0)

        elif emotions_result.Emotion[i] == 'fear' or emotions_result.Emotion[i] == 'surprise':
            # print emotions_result.Emotion[i]
            happy_p.append(0)
            sad_p.append(0)
            angry_p.append(0)
            fear_p.append(emotions_result.Emotion[i])
            neutral_p.append(0)

        else:
            # print emotions_result.Emotion[i]
            happy_p.append(0)
            sad_p.append(0)
            angry_p.append(0)
            fear_p.append(0)
            neutral_p.append(emotions_result.Emotion[i])

    print happy_p, len(happy_p)

    print len(happy_p), len(sad_p),len(angry_p),len(fear_p), len(neutral)
    os._exit(0)
    # fig, ax = plt.subplots()
    # ax.hist(happy_p, bins=25, alpha=0.8, normed=True)
    # ax.hist(sad_p, bins=25, alpha=0.8, normed=True)
    # ax.hist(angry_p, bins=25, alpha=0.8, normed=True)
    # ax.hist(fear_p, bins=25, alpha=0.8, normed=True)
    # ax.hist(neutral_p, bins=25, alpha=0.8, normed=True)

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())


    # sub_axix = filter(lambda x:x%200 == 0, len(emotions_result.id))
    plt.title('Result Analysis')
    plt.plot(emotions_result.id, happy_p, color='green', label='training accuracy')
    plt.plot(emotions_result.id, sad_p, color='red', label='testing accuracy')
    plt.plot(emotions_result.id, angry_p,  color='skyblue', label='PN distance')
    plt.plot(emotions_result.id, fear_p, color='blue', label='threshold')
    plt.plot(emotions_result.id, neutral_p, color='blue', label='threshold')
    plt.legend() 
     
    plt.xlabel('iteration times')
    plt.ylabel('rate')

    ax.set_title("each emotion probability values")
    plt.show()


if __name__ == '__main__':
    # single one person's info
    # single_people_number(user=hudi, n_groups=4)
    file_path = './'
    csv_file = file_path + 'hudi_1.csv'
    emotions_result = pd.read_csv(csv_file)

    (happy,sad,angry,fear,neutral) = get_single_emotion(emotions_result)
    (happy_p,sad_p,angry_p,fear_p,neutral_p) = get_single_probability(emotions_result)
    # print happy[0],happy_p[0]
    # print len(happy_p), len(sad_p), len(angry_p), len(fear_p), len(neutral_p)
    sigle_proability(emotions_result)

    os._exit(0)

    for i in range(0, len(emotions_result.id)):
        row = emotions_result.iloc[i]['Time']
        # print row

        # time_text.append(time_text)

        os._exit(0)

    # for i in range(0, len(emotions_result.Time)):
    #     row = emotions_result.iloc[i]['Time']
    #     # print row
    #     a = int(row)

    #     time_text = datetime.utcfromtimestamp(a)
    #     time_text = time_text.strftime("%H:%M:%S")
    #     print time_text
    #     dates = matplotlib.dates.datestr2num(time_text)
    #     print dates
    #     # time_text.append(time_text)
    #     os._exit(0)




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

