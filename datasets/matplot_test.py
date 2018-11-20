# # import numpy as np
# # import os
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates
# # from mpl_toolkits.axes_grid1 import *
# # from mpl_toolkits.axes_grid1 import make_axes_locatable
# # from mpl_toolkits.axes_grid1 import host_subplot
# # import mpl_toolkits.axisartist as AA
# # from dataset_chart import dataset_number
# # from dataset_chart import three_way_plot

# # class PointBrowser(object):
# #     """
# #     Click on a point to select and highlight it -- the data that
# #     generated the point will be shown in the lower axes.  Use the 'n'
# #     and 'p' keys to browse through the next and previous points
# #     """

# #     def __init__(self):
# #         self.lastind = 0

# #         self.text = ax.text(0.05, 0.95, 'selected: none',
# #                             transform=ax.transAxes, va='top')
# #         self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
# #                                  color='yellow', visible=False)

# #     def onpress(self, event):
# #         if self.lastind is None:
# #             return
# #         if event.key not in ('n', 'p'):
# #             return
# #         if event.key == 'n':
# #             inc = 1
# #         else:
# #             inc = -1

# #         self.lastind += inc
# #         self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
# #         self.update()

# #     def onpick(self, event):

# #         if event.artist != line:
# #             return True

# #         N = len(event.ind)
# #         if not N:
# #             return True

# #         # the click locations
# #         x = event.mouseevent.xdata
# #         y = event.mouseevent.ydata

# #         distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
# #         indmin = distances.argmin()
# #         dataind = event.ind[indmin]

# #         self.lastind = dataind
# #         self.update()

# #     def update(self):
# #         if self.lastind is None:
# #             return

# #         dataind = self.lastind

# #         ax2.cla()
# #         ax2.plot(X[dataind])

# #         ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
# #                  transform=ax2.transAxes, va='top')
# #         ax2.set_ylim(-0.5, 1.5)
# #         self.selected.set_visible(True)
# #         self.selected.set_data(xs[dataind], ys[dataind])

# #         self.text.set_text('selected: %d' % dataind)
# #         fig.canvas.draw()


# # if __name__ == '__main__':
# #     import matplotlib.pyplot as plt
# #     # Fixing random state for reproducibility
# #     np.random.seed(19680801)

# #     X = np.random.rand(100, 200)
# #     xs = np.mean(X, axis=1)
# #     ys = np.std(X, axis=1)

# #     index = np.arange(13)

# #     network = ['AlexNet', 'BOVW+VGG(4x)', 
# #             'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
# #             'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

# #     network_our = [ 0, 'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
# #                 'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

# #     accuracy = [0,71.20, 75.42, 68.61, 67.76, 65.67, 67.65, 69.24, 70.13, 69.57, 66.48, 70.13, 70.19]
# #     # std=0.15
# #     accuracy_std = [0, 0, 68.46, 67.61, 65.52, 67.50, 69.09, 69.98, 69.42, 66.33, 69.98, 70.04] 
# #     F1_score = [0, 0, 68.53, 67.84, 65.60, 67.57, 69.11, 70.00, 69.41, 65.98, 70.04, 70.13]

# #     # model's parameters
# #     infer_time = [ 0, 0, 2000, 730, 597, 1000, 2000, 2000, 1000, 470, 832, 701]
# #     parameters = [0, 0, 20875248, 3235463, 973135, 2284615, 2284615, 
# #                 6671671, 264679, 58423, 290719, 290719]
# #     memory = [0, 0, 250.70, 39.00, 12.40, 27.90, 27.90, 80.30, 3.90, 0.898, 3.90, 3.90]


# #     fig, (ax, ax2) = plt.subplots(2, 1)
# #     ax.set_title('click on point to plot time series')
# #     line, = ax.plot(index, accuracy, 'o')  # 5 points tolerance

# #     browser = PointBrowser()

# #     fig.canvas.mpl_connect('pick_event', browser.onpick)
# #     fig.canvas.mpl_connect('key_press_event', browser.onpress)

# #     plt.show()


# # import matplotlib.pyplot as plt
# # from matplotlib.lines import Line2D
# # from matplotlib.patches import Rectangle
# # from matplotlib.text import Text
# # from matplotlib.image import AxesImage
# # import numpy as np
# # from numpy.random import rand

# # if 1:  # simple picking, lines, rectangles and text
# #     fig, (ax1, ax2) = plt.subplots(2, 1)
# #     ax1.set_title('click on points, rectangles or text', picker=True)
# #     ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
# #     line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance

# #     # pick the rectangle
# #     bars = ax2.bar(range(10), rand(10), picker=True)
# #     for label in ax2.get_xticklabels():  # make the xtick labels pickable
# #         label.set_picker(True)

# #     def onpick1(event):
# #         if isinstance(event.artist, Line2D):
# #             thisline = event.artist
# #             xdata = thisline.get_xdata()
# #             ydata = thisline.get_ydata()
# #             ind = event.ind
# #             print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
# #         elif isinstance(event.artist, Rectangle):
# #             patch = event.artist
# #             print('onpick1 patch:', patch.get_path())
# #         elif isinstance(event.artist, Text):
# #             text = event.artist
# #             print('onpick1 text:', text.get_text())

# #     fig.canvas.mpl_connect('pick_event', onpick1)



# # plt.show()

# import matplotlib.pyplot as plt
# import numpy as np


# def addtext(ax, props):
#     ax.text(0.5, 0.5, 'text 0', props, rotation=0)
#     ax.text(1.5, 0.5, 'text 45', props, rotation=45)
#     ax.text(2.5, 0.5, 'text 135', props, rotation=135)
#     ax.text(3.5, 0.5, 'text 225', props, rotation=225)
#     ax.text(4.5, 0.5, 'text -45', props, rotation=-45)
#     for x in range(0, 5):
#         ax.scatter(x + 0.5, 0.5, color='r', alpha=0.5)
#     ax.set_yticks([0, .5, 1])
#     ax.set_xlim(0, 5)
#     ax.grid(True)


# # the text bounding box
# bbox = {'fc': '0.8', 'pad': 0}

# fig, axs = plt.subplots(2, 1)

# addtext(axs[0], {'ha': 'center', 'va': 'center', 'bbox': bbox})
# axs[0].set_xticks(np.arange(0, 5.1, 0.5), [])
# axs[0].set_ylabel('center / center')

# addtext(axs[1], {'ha': 'left', 'va': 'bottom', 'bbox': bbox})
# axs[1].set_xticks(np.arange(0, 5.1, 0.5))
# axs[1].set_ylabel('left / bottom')

# plt.show()



# # simple picking, lines, rectangles and text
# def two_scatter_bar(index, bar_value, scatter_value, x_label): 
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     bar_width = 0.5

#     # pick the rectangle
#     line= ax1.scatter(index[2:12], scatter_value[2:12], s=[2000, 1700, 600, 1200, 1200, 1700, 300, 140, 400, 400],
#                         c = ['#996699','#CC99CC','#9999CC',
#                     '#FFFFCC', '#FFFF99', '#FFCC33',
#                     '#FF9933','#CC9900','#FF9900', 'g']) 

#     ax1.annotate("290,719",xy=('8', 1500000), xytext=('7.5', 4000000), fontsize=12,
#                     arrowprops=dict(facecolor='red', width=2))

#     bars = ax2.bar(index, bar_value, bar_width, 
#         color= ['#660033','#993366','#996699','#CC99CC','#9999CC',
#                 '#FFFFCC', '#FFFF99', '#FFCC33',
#                 '#FF9933','#CC9900','#FF9900', 'g'])

#     # ax1.legend([2000, 1500, 500,  100], ['350MB', '200MB', '150MB', []]) 

#     ax1.set_ylabel("Network's parameters", picker=True, fontsize=13)
#     # ax1.set_xlabel('Neutral Network', fontsize=20)
#     ax2.set_ylabel('accuracy', picker=True)

#     ax1.set_xticks(index[2:12])
#     ax1.set_xticklabels(network[2:12],rotation=35, fontsize=7)

#     ax2.set_xticks(index+bar_width)
#     ax2.set_xticklabels(network,rotation=25, fontsize=10)
   
#     ax1.set_yticks([2000000,4000000,6000000,8000000,1000000,12000000,14000000,16000000,
#                 18000000,20000000,22000000,24000000,26000000])
#     # ax2.set_ylim(0, max(bar_value)+20 ) 
#     ax2.set_yticks([10,20,30,40,50,60,70,75,85,95])

#     def onpick1(event):
#         if isinstance(event.artist, Line2D):
#             thisline = event.artist
#             xdata = thisline.get_xdata()
#             ydata = thisline.get_ydata()
#             ind = event.ind
#             print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
#         elif isinstance(event.artist, Rectangle):
#             patch = event.artist
#             print('onpick1 patch:', patch.get_path())
#         elif isinstance(event.artist, Text):
#             text = event.artist
#             print('onpick1 text:', text.get_text())

#     fig.canvas.mpl_connect('pick_event', onpick1)
#     plt.grid(True)
#     plt.show()


from matplotlib import pyplot as plt
import numpy as np
randn = np.random.randn
from pandas import *
idx = Index(np.arange(1,7))
df = DataFrame(randn(6, 4), index=idx, columns=['A', 'B', 'C', 'D'])
vals = np.around(df.values,2)
fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])
the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, 
    olWidths = [0.1]*vals.shape[1], loc='center',cellLoc='center')
the_table.set_fontsize(20)

the_table.scale(2.5,2.58)
