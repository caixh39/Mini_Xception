""" plot concat_xception result
"""
import os
import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import *
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def dataset_number(dataset):
	emotions = np.zeros(7)
	datasets_path = dataset + '/'
	csv_file = os.path.join(datasets_path, dataset + '.csv')
	# train_csv = os.path.join(datasets_path, 'train.csv')
	# val_csv = os.path.join(datasets_path, 'val.csv')
	# test_csv = os.path.join(datasets_path, 'test.csv')
	data = pd.read_csv(csv_file , dtype='a')
	label = np.array(data['emotion'])
	img_data = np.array(data['pixels'])
	N_sample = label.size
	for i in label:
		for j in range(7):
			if int(i) == j:
				emotions[j] = emotions[j] + 1
	# print(emotions)
	return emotions

def bar_same_figure(bar1_value, bar2_value):
	# figure plot 
	fig, ax = plt.subplots()
	index = np.arange(7)
	bar_width = 0.35

	jaffe = ax.bar(index,emotions_jaf, bar_width, color= 'r', label='jaffer')
	fer2013 = ax.bar(index+bar_width,emotions_fer, bar_width, color= 'b', label='fer2013')
	ax.set_title("Facial expression dataset's emotion")
	ax.set_xlabel('emotions')
	ax.set_ylabel('number')
	ax.set_xticks(index + bar_width / 2)
	ax.set_xticklabels(('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'))
	ax.legend()

	fig.tight_layout()
	plt.grid()
	plt.show()

# describe the dataset emotion's number
def dataset_emotion_number(bar_value, dataset_name):
	""" """
	bar_number = 7
	bar_width = 0.5
	index = np.arange(bar_number) 
	plt.bar(index, bar_value, bar_width, 
			color= ['#CC3300','#339966','#0066FF','#FFFF00','#663366','#CCCCCC','#006699'])
	# plt.text('4.8', 36, r'JAFFE dataset',fontdict={'size': 17, 'color': '#000000'})
	plt.text('4.5', 9200, r'Fer2013 dataset',fontdict={'size': 17, 'color': '#000000'})
	plt.xlabel('emotions',fontsize=20)
	plt.xticks(index, ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'], rotation=0)
	plt.ylabel('number', fontsize=20)
	# plt.yticks([5,10,15,20,25,30,35,40])
	plt.yticks([1000,2000,3000,4000,5000,6000,7000,8000, 9000, 10000])
	plt.legend(color='#000000',loc="upper right")
	plt.grid()
	plt.show()


def plot_accuracy(network, accuracy):
	""" """
	bar_number = 12
	bar_width = 0.7
	bar_value = accuracy
	index = np.arange(bar_number) 
	plt.bar(index, bar_value, bar_width, 
			color= ['#666666','#666666',
					'#660033','#993366','#996699','#CC99CC','#9999CC',
					'#FFFFCC', '#FFFF99', '#FFCC33',
					'#FF9933','#CC9900','#FF9900'])
	plt.annotate('70.13%',xy=('10.3', 70), xytext=('8.0', 78), fontsize=17,
					arrowprops=dict(facecolor='red', width=5))
	plt.xlabel('Neutral Network',fontsize=20)
	plt.xticks(index, network, rotation=35)
	plt.ylabel('test accuracy', fontsize=20)
	plt.yticks([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
	plt.legend(color='#000000',loc="upper right")
	plt.grid()
	plt.show()


def plot_f1_score(network, accuracy):
	""" """
	bar_number = 10
	bar_width = 0.7
	bar_value = accuracy[2:12]
	index = np.arange(bar_number) 
	plt.bar(index, bar_value, bar_width, 
			color= ['#660033','#993366','#996699','#CC99CC','#9999CC',
					'#FFFFCC', '#FFFF99', '#FFCC33',
					'#FF9933','#CC9900','#FF9900'])
	plt.annotate('70.04%',xy=('10.3', 70), xytext=('8.0', 78), fontsize=17,
					arrowprops=dict(facecolor='red', width=5))
	plt.xlabel('Neutral Network',fontsize=20)
	plt.xticks(index, network, rotation=35)
	plt.ylabel('F1-Score', fontsize=20)
	plt.yticks([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
	plt.legend(color='#000000',loc="upper right")
	plt.grid()
	plt.show()


def bar_accuracy_f1score(network, bar1_accuracy, bar2_f1score):
	# figure plot 
	fig, ax = plt.subplots()
	index = np.arange(12)
	bar_width = 0.35

	accuracy = ax.bar(index, bar1_accuracy, bar_width, color= '#666666', label='Accuracy')
	f1_score = ax.bar(index+bar_width, bar2_f1score, bar_width, color= '#CC9999', label='F1-Score')

	plt.annotate('70.13%',xy=('10.3', 70), xytext=('8.5', 78), fontsize=15,
					arrowprops=dict(facecolor='red', width=5))

	plt.annotate('70.04%',xy=('10.5', 70), xytext=('11', 78), fontsize=15,
					arrowprops=dict(facecolor='red', width=5))

	# ax.set_title("Facial expression dataset's emotion")
	ax.set_xlabel('Neutral Network')
	# ax.set_ylabel('number')
	# ax.set_xticks(index, network)
	ax.set_xticks(index+bar_width)
	ax.set_xticklabels(network,rotation=45)
	ax.set_yticks([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
	plt.legend(loc="upper left")
	plt.grid()
	plt.show()


def scatter_parameters(network, parameters):
	# figure plot 
	fig, ax = plt.subplots()
	index = np.arange(10)
	bar_width = 0.35

	plt.scatter(index, parameters, s=[2000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400], 
			c = ['#660033','#993366','#996699','#CC99CC','#9999CC',
				'#FFFFCC', '#FFFF99','#FF9933','#CC9900','#FF9900'])

	plt.annotate("290,719",xy=('8', 1500000), xytext=('7.5', 4000000), fontsize=17,
					arrowprops=dict(facecolor='red', width=5))
	plt.xlabel('Neutral Network',fontsize=20)
	plt.xticks(index, network[2:12], rotation=35)
	plt.ylabel("Network's parameters", fontsize=20)
	plt.yticks([2000000,4000000,6000000,8000000,1000000,12000000,14000000,16000000,
				18000000,20000000,22000000,24000000,26000000])
	# plt.legend(color='#000000',loc="upper right")
	plt.legend()
	plt.grid()
	fig.tight_layout()
	plt.show()

def scatter_infer_time(network, memory):
	# figure plot 
	fig, ax = plt.subplots()
	index = np.arange(10)
	bar_width = 0.35

	plt.scatter(index, memory, s=[2000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400], 
			c = ['#660033','#993366','#996699','#CC99CC','#9999CC',
				'#FFFFCC', '#FFFF99','#FF9933','#CC9900','#FF9900'])

	plt.annotate("83.2",xy=('8', 1500000), xytext=('7.5', 4000000), fontsize=17,
					arrowprops=dict(facecolor='red', width=5))
	plt.xlabel('Neutral Network',fontsize=20)
	plt.xticks(index, network[2:12], rotation=35)
	plt.ylabel("Network's parameters", fontsize=20)
	plt.yticks([2000000,4000000,6000000,8000000,1000000,12000000,14000000,16000000,
				18000000,20000000,22000000,24000000,26000000])
	# plt.legend(color='#000000',loc="upper right")
	plt.legend()
	plt.grid()
	fig.tight_layout()
	plt.show()


def memory_infer_time(network, infer_time, parameters):
	"""figure plot """
	number = 11
	index = np.arange(number)

	host = host_subplot(111, axes_class=AA.Axes)
	# host = subplots()
	par = host.twinx()

	host.set_ylim(0, 2500)

	host.set_xlabel("Neutral Network", fontsize=50)
	host.xaxis.set_ticks(index, network[1:12])
	host.set_ylabel("Forward time per image [x10 us]", fontsize=35)
	par.set_ylabel("Parameters", fontsize=50)

	p1, = host.plot(index, infer_time[1:12], "ro-", label="Time")
	p2, = par.plot(index, parameters[1:12], "b^-",label="Parameters")

	leg = plt.legend()

	host.yaxis.get_label().set_color(p1.get_color())
	leg.texts[0].set_color(p1.get_color())

	par.yaxis.get_label().set_color(p2.get_color())
	leg.texts[1].set_color(p2.get_color())

	plt.grid()
	plt.show()

	
def make_patch_spines_invisible(ax): 
    ax.set_frame_on(True) 
    ax.patch.set_visible(False) 
    for sp in ax.spines.itervalues(): 
    	sp.set_visible(False) 


def three_way_plot(index,time, memory, parameters, network, category): 
    fig, host = plt.subplots() 
    fig.subplots_adjust(right=0.6) 
    par1 = host.twinx() 
    par2 = host.twinx() 

    par2.spines["right"].set_position(("axes", 1.1)) 

    p1, = host.plot(index, time, "b", marker='s', linewidth=1.2, markersize=5, label="Time") 
    p2, = par1.plot(index, memory, "r", marker='^', linewidth=1.4, linestyle='dashed',markersize=6, label="Memory") 
    p3, = par2.plot(index, parameters, "g", marker='+', linewidth=1.4, linestyle='dashed',markersize=6,  label="Parameters") 

    par1.annotate("0.898MB", xy=('8', 8), xytext=('8', 8), fontsize=11, color = 'red')
    par1.annotate("1.3MB", xy=('9', 8), xytext=('9', 8), fontsize=11, color = 'red')
    par1.annotate("3.9MB", xy=('10', 8), xytext=('10', 8), fontsize=11, color = 'red')
    par1.annotate("3.9MB", xy=('11', 8), xytext=('11', 8), fontsize=11, color = 'red')

    host.set_xlim(0, max(index)+1) 
    # host.set_ylim(min(time), max(time) + 2) 
    host.set_yticks([2,4,6,8,10,12,14])
    par1.set_ylim(0, max(memory)+ 15 ) 
    par2.set_ylim(0, max(parameters) + 8500000) 

    host.set_xlabel("Neutral Network", fontsize=14) 
    host.set_ylabel("Forward time per image [ms]", fontsize=14) 
    par1.set_ylabel("Memory [MB]", fontsize=14) 
    par2.set_ylabel("Parameters", fontsize=14) 

    host.yaxis.label.set_color(p1.get_color()) 
    par1.yaxis.label.set_color(p2.get_color()) 
    par2.yaxis.label.set_color(p3.get_color()) 

    lines = [p1, p2, p3] 
    labels = network

    start, end, step_size = 1, len(network)+1, 1 

    # host.set_xticks(np.arange(start, end, step_size)) 
    host.set_xticks(index)
    host.set_xticklabels(labels, rotation=60, fontsize=11) 

    host.legend(lines, [l.get_label() for l in lines]) 
    plt.tight_layout() 
    plt.grid(ls='--')
    plt.show() 


def addtext(ax, props):
    ax.text(0.5, 0.5, 'text 0', props, rotation=0)
    ax.text(1.5, 0.5, 'text 45', props, rotation=0)
    ax.text(2.5, 0.5, 'text 135', props, rotation=0)
    ax.text(3.5, 0.5, 'text 225', props, rotation=0)
    ax.text(4.5, 0.5, 'text -45', props, rotation=0)
    for x in range(0, 5):
        ax.scatter(x + 0.5, 0.5, color='r', alpha=0.5)
    ax.set_yticks([0, .5, 1])
    ax.set_xlim(0, 5)
    ax.grid(True)

# simple picking, lines, rectangles and text
def two_scatter_bar(index, accuracy, kappa, colors, scatter, network): 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    bar_width = 0.5

    # pick the rectangle
    # for x in range(0, index):
    ax1.scatter(index[2:13], kappa[2:13], s=scatter[2:13],
				color = colors[2:13]) 
    ax1.scatter(index[2:13], kappa[2:13], s=8, color = '#000000') 

    ax1.annotate("20,875,247", xy=('2', 0.66), xytext=('1.4', 0.81), fontsize=11, color = 'blue')
    ax1.annotate("3,235,463", xy=('2.9', 0.65), xytext=('3.1', 0.7), fontsize=11)
    ax1.annotate("973,135", xy=('4', 0.62), xytext=('3.6', 0.625), fontsize=11)
    ax1.annotate("2,284,615", xy=('5', 0.65), xytext=('4.5', 0.685), fontsize=11)
    ax1.annotate("4,025,483", xy=('7', 0.68), xytext=('6.4', 0.7), fontsize=11)
    ax1.annotate("2,284,615", xy=('8', 0.63), xytext=('7.5', 0.755), fontsize=11)
    ax1.annotate("6,671,671", xy=('6', 0.67), xytext=('5.6', 0.75), fontsize=11)
    ax1.annotate("58,423", xy=('7.5', 0.62), xytext=('8.75', 0.615), fontsize=11,color = 'blue')
    ax1.annotate("88,999", xy=('10', 0.64), xytext=('9.6', 0.63), fontsize=11)
    ax1.annotate("290,399", xy=('11.2', 0.66), xytext=('10.6', 0.68), fontsize=11, color = 'blue')
    ax1.annotate("264,679", xy=('11.2', 0.66), xytext=('11.7', 0.665), fontsize=11)

    # [3000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400, 400]
    # ax1.annotate("20,875,247", xy=('2.5', 22000000), xytext=('2.5', 22000000), fontsize=11)
    # ax1.annotate("3,235,247", xy=('2.9', 7000000), xytext=('2.7', 7000000), fontsize=11)
    # ax1.annotate("973,135", xy=('4', 1000000), xytext=('3.5', 4000000), fontsize=11)
    # ax1.annotate("2,284,615", xy=('5', 5000000), xytext=('4.4', 10000000), fontsize=11)
    # ax1.annotate("2,284,615", xy=('6', 5000000), xytext=('5.3', 6000000), fontsize=11)
    # ax1.annotate("6,671,671", xy=('7', 9000000), xytext=('6.6', 11000000), fontsize=11)
    # ax1.annotate("264,679", xy=('8', 1000000), xytext=('7.2', 1900000), fontsize=11)

    # ax1.annotate("58,423", xy=('7.5', 1000000), xytext=('8.5', 1900000), fontsize=11,
    # 			color = 'blue')
    # ax1.annotate("290,719", xy=('10', 1000000), xytext=('9.4', 1900000), fontsize=11)
    # ax1.annotate("290,719", xy=('11.2', 1000000), xytext=('10.7', 1900000), fontsize=11)


    bars = ax2.bar(index, accuracy, bar_width, color=colors)
    ax2.annotate("71.20", xy=('0', 0.66), xytext=('0', 72), fontsize=10)
    ax2.annotate("75.42", xy=('0', 0.66), xytext=('1', 76), fontsize=10)

    ax2.annotate("68.51", xy=('0', 0.66), xytext=('2', 69), fontsize=10)
    ax2.annotate("67.76", xy=('0', 0.66), xytext=('3', 68.3), fontsize=10)
    ax2.annotate("65.67", xy=('0', 0.66), xytext=('4', 66), fontsize=10)

    ax2.annotate("67.65", xy=('0', 0.66), xytext=('5', 68), fontsize=10)
    ax2.annotate("67.18", xy=('0', 0.66), xytext=('6', 67.5), fontsize=10)
    ax2.annotate("69.24", xy=('0', 0.66), xytext=('7', 70), fontsize=10)
    ax2.annotate("70.13", xy=('0', 0.66), xytext=('8', 70.5), fontsize=10)

    ax2.annotate("66.48", xy=('0', 0.66), xytext=('9', 67.3), fontsize=10)
    ax2.annotate("66.82", xy=('0', 0.66), xytext=('10', 67.5), fontsize=10)
    ax2.annotate("70.13", xy=('0', 0.66), xytext=('11', 71), fontsize=10)
    ax2.annotate("69.57", xy=('0', 0.66), xytext=('12', 70.3), fontsize=10)

    ax1.set_ylabel("Cohen Kappa", picker=True, fontsize=14)
    ax2.set_xlabel('Neutral Network', fontsize=20)
    ax2.set_ylabel('Accuracy', picker=True, fontsize=14)

    ax1.set_xticks(index[2:13])
    ax1.set_xticklabels(network[2:13],rotation=35, fontsize=11)

    ax2.set_xticks(index)
    ax2.set_xticklabels(network,rotation=25, fontsize=10)
   
    # ax1.set_yticks([1000000,4000000,8000000,1000000,12000000,14000000,16000000,
				# 18000000,20000000,22000000,26000000])

    ax2.set_yticks([10,20,30,40,50,60,70,75,85,95])

    ax1.grid(True)
    ax2.grid(True)
    plt.show()


def addtext(ax, props):
    ax.text(0.5, 0.5, 'text 0', props, rotation=0)
    ax.text(1.5, 0.5, 'text 45', props, rotation=45)
    ax.text(2.5, 0.5, 'text 135', props, rotation=135)
    ax.text(3.5, 0.5, 'text 225', props, rotation=225)
    ax.text(4.5, 0.5, 'text -45', props, rotation=-45)
    for x in range(0, 5):
        ax.scatter(x + 0.5, 0.5, color='r', alpha=0.5)
    ax.set_yticks([0, .5, 1])
    ax.set_xlim(0, 5)
    ax.grid(True)



if __name__ == '__main__':

	dataset = ['fer2013','jaffe']
	file_path = './model_result/'
	csv_file = file_path + 'result_inter.csv'
	emotions_result = pd.read_csv(csv_file)

	network = emotions_result.network1
	accuracy = emotions_result.accuracy
	kappa = emotions_result.kappa
	memory = emotions_result.memory
	parameters = emotions_result.parameters
	time = emotions_result.time
	colors = emotions_result.colors
	scatter = emotions_result.scatter

	# getting datasets number
	emotions_fer = dataset_number(dataset[0])
	emotions_jaf = dataset_number(dataset[1])

	index = np.arange(12)

	# bar_same_figure(emotions_fer, emotions_jaf)

	# drawBarChartPoseRatio()
	# dataset_emotion_number(emotions_fer, 'Fer2013')
	# plot_accuracy(network, accuracy)
	# plot_f1_score(network, accuracy, F1_score)
	# bar_accuracy_f1score(network, accuracy, F1_score)
	# scatter_parameters(network, memory)
	# memory_infer_time(network, infer_time, parameters)


	three_way_plot(index, time[1:13], memory[1:13], parameters[1:13], network[1:13], "category") 

	# two_scatter_bar(index, accuracy, kappa, colors, scatter,network)
