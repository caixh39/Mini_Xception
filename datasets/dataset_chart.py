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


# the best result matrix:70.13%
matrix = [[311, 39, 11, 73, 5, 48],
 		   [ 12, 38, 1, 1, 1, 1, 1],
  		   [ 74, 2, 264, 18, 91, 39, 40],
  		   [ 23, 0, 11, 785, 24, 9, 27],
  		   [ 60, 1, 56, 29, 346, 6, 96],
  		   [ 10, 1, 33, 20, 7, 336, 9],
  		   [ 31, 2, 34, 30, 87, 5, 437]]



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


def three_way_plot(xs,ys,ws,zs,network,category): 
    fig, host = plt.subplots() 
    fig.subplots_adjust(right=0.6) 
    par1 = host.twinx() 
    par2 = host.twinx() 

    par2.spines["right"].set_position(("axes", 1.1)) 

    p1, = host.plot(xs, ys, "b", marker='s', linewidth=1.2, markersize=5, label="Time") 
    p2, = par1.plot(xs, ws, "r", marker='^', linewidth=1.4, linestyle='dashed',markersize=6, label="Parameters") 
    p3, = par2.plot(xs, zs, "g", marker='+', linewidth=1.4, linestyle='dashed',markersize=6,  label="Memory") 

    # p2, = host.scatter(xs, ws, s=[2000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400], 
				# 			c = ['#660033','#993366','#996699','#CC99CC','#9999CC',
				# 			'#FFFFCC', '#FFFF99','#FF9933','#CC9900','#FF9900'], label="Parameters")
    # host.annotate("8",xy=('8', 33), xytext=('7.5', 4000000), fontsize=17,
				#  	arrowprops=dict(facecolor='red', width=5))

    host.set_xlim(0, max(xs)+1) 
    host.set_ylim(min(ys), max(ys) + 1200) 
    par1.set_ylim(0, max(ws)+ 900000 ) 
    par2.set_ylim(0, max(zs) + 70) 

    host.set_xlabel("Neutral Network", fontsize=14) 
    host.set_ylabel("Forward time per image [x10 us]", fontsize=14) 
    par1.set_ylabel("Parameters", fontsize=14) 
    par2.set_ylabel("Memory [MB]", fontsize=14) 

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
    plt.grid()
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
def two_scatter_bar(index, bar_value, scatter_value, x_label): 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    bar_width = 0.5

    # pick the rectangle
    # for x in range(0, index):
    ax1.scatter(index[2:12], scatter_value[2:12], s=[2000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400],
				color = ['#996699','#CC99CC','#9999CC',
				'#FFFFCC', '#FFFF99', '#FFCC33',
				'#FF9933','#CC9900','#FF9900', 'g']) 
    ax1.scatter(index[2:12], scatter_value[2:12], s=15, color = '#000000') 

    ax1.annotate("20,875,247", xy=('2.5', 22000000), xytext=('2.5', 22000000), fontsize=11)
    ax1.annotate("3,235,247", xy=('2.9', 7000000), xytext=('2.7', 7000000), fontsize=11)
    ax1.annotate("973,135", xy=('4', 1000000), xytext=('3.5', 4000000), fontsize=11)
    ax1.annotate("2,284,615", xy=('5', 5000000), xytext=('4.4', 10000000), fontsize=11)
    ax1.annotate("2,284,615", xy=('6', 5000000), xytext=('5.3', 6000000), fontsize=11)
    ax1.annotate("6,671,671", xy=('7', 9000000), xytext=('6.6', 11000000), fontsize=11)
    ax1.annotate("264,679", xy=('8', 1000000), xytext=('7.2', 1900000), fontsize=11)

    ax1.annotate("58,423", xy=('7.5', 1000000), xytext=('8.5', 1900000), fontsize=11,
    			color = 'blue')
    ax1.annotate("290,719", xy=('10', 1000000), xytext=('9.4', 1900000), fontsize=11)
    ax1.annotate("290,719", xy=('11.2', 1000000), xytext=('10.7', 1900000), fontsize=11)


    bars = ax2.bar(index, bar_value, bar_width, 
		color= ['#660033','#993366','#996699','#CC99CC','#9999CC',
				'#FFFFCC', '#FFFF99', '#FFCC33',
				'#FF9933','#CC9900','#FF9900', 'g'])

    ax1.set_ylabel("Network's parameters", picker=True, fontsize=13)
    ax2.set_xlabel('Neutral Network', fontsize=20)
    ax2.set_ylabel('accuracy', picker=True, fontsize=13)

    ax1.set_xticks(index[2:12])
    ax1.set_xticklabels(network[2:12],rotation=35, fontsize=11)

    ax2.set_xticks(index)
    ax2.set_xticklabels(network,rotation=25, fontsize=10)
   
    ax1.set_yticks([1000000,4000000,8000000,1000000,12000000,14000000,16000000,
				18000000,20000000,22000000,26000000])
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

	dataset = ['Fer2013','JAFFE']
	emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

	# concat_Xception result: name, time, memory, accuracy
	network = ['AlexNet', 'BOVW+VGG', 
				'Xception', 'Mobile', 'Shuffle', 'MobileV2', 'MobileV2(96)','MobileV2(96,C)',
				'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

	network_our = [ 0, 'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
				'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

	accuracy = [71.20, 75.42, 68.61, 67.76, 65.67, 67.65, 69.24, 70.13, 69.57, 66.48, 70.13, 70.19]
	# std=0.15
	accuracy_std = [0, 0, 68.46, 67.61, 65.52, 67.50, 69.09, 69.98, 69.42, 66.33, 69.98, 70.04] 
	F1_score = [0, 0, 68.53, 67.84, 65.60, 67.57, 69.11, 70.00, 69.41, 65.98, 70.04, 70.13]

	# model's parameters
	infer_time = [ 0, 0, 2000, 730, 597, 1000, 2000, 2000, 1000, 470, 832, 701]
	parameters = [0, 0, 20875248, 3235463, 973135, 2284615, 2284615, 
				6671671, 264679, 58423, 290719, 290719]
	memory = [0, 0, 250.70, 39.00, 12.40, 27.90, 27.90, 80.30, 3.90, 0.898, 3.90, 3.90]

	color= ['#660033','#993366','#996699','#CC99CC','#9999CC',
		'#FFFFCC', '#FFFF99', '#FFCC33',
		'#FF9933','#CC9900','#FF9900', 'g']

	# getting datasets number
	emotions_fer = dataset_number('fer2013')
	emotions_jaf = dataset_number('jaffe')

	index = np.arange(12)

	# bar_same_figure(emotions_fer, emotions_jaf)

	# drawBarChartPoseRatio()
	# dataset_emotion_number(emotions_fer, 'Fer2013')
	# plot_accuracy(network, accuracy)
	# plot_f1_score(network, accuracy, F1_score)
	# bar_accuracy_f1score(network, accuracy, F1_score)
	# scatter_parameters(network, memory)
	# memory_infer_time(network, infer_time, parameters)


	# three_way_plot(index, infer_time[1:12], parameters[1:12], memory[1:12], network_our, "category") 

	two_scatter_bar(index,accuracy, parameters, network)
