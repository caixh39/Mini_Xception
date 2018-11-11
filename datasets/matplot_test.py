# import numpy as np
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
# from dataset_chart import dataset_number
# from dataset_chart import three_way_plot

# class PointBrowser(object):
#     """
#     Click on a point to select and highlight it -- the data that
#     generated the point will be shown in the lower axes.  Use the 'n'
#     and 'p' keys to browse through the next and previous points
#     """

#     def __init__(self):
#         self.lastind = 0

#         self.text = ax.text(0.05, 0.95, 'selected: none',
#                             transform=ax.transAxes, va='top')
#         self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
#                                  color='yellow', visible=False)

#     def onpress(self, event):
#         if self.lastind is None:
#             return
#         if event.key not in ('n', 'p'):
#             return
#         if event.key == 'n':
#             inc = 1
#         else:
#             inc = -1

#         self.lastind += inc
#         self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
#         self.update()

#     def onpick(self, event):

#         if event.artist != line:
#             return True

#         N = len(event.ind)
#         if not N:
#             return True

#         # the click locations
#         x = event.mouseevent.xdata
#         y = event.mouseevent.ydata

#         distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
#         indmin = distances.argmin()
#         dataind = event.ind[indmin]

#         self.lastind = dataind
#         self.update()

#     def update(self):
#         if self.lastind is None:
#             return

#         dataind = self.lastind

#         ax2.cla()
#         ax2.plot(X[dataind])

#         ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
#                  transform=ax2.transAxes, va='top')
#         ax2.set_ylim(-0.5, 1.5)
#         self.selected.set_visible(True)
#         self.selected.set_data(xs[dataind], ys[dataind])

#         self.text.set_text('selected: %d' % dataind)
#         fig.canvas.draw()


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     # Fixing random state for reproducibility
#     np.random.seed(19680801)

#     X = np.random.rand(100, 200)
#     xs = np.mean(X, axis=1)
#     ys = np.std(X, axis=1)

#     index = np.arange(13)

#     network = ['AlexNet', 'BOVW+VGG(4x)', 
#             'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
#             'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

#     network_our = [ 0, 'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
#                 'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

#     accuracy = [0,71.20, 75.42, 68.61, 67.76, 65.67, 67.65, 69.24, 70.13, 69.57, 66.48, 70.13, 70.19]
#     # std=0.15
#     accuracy_std = [0, 0, 68.46, 67.61, 65.52, 67.50, 69.09, 69.98, 69.42, 66.33, 69.98, 70.04] 
#     F1_score = [0, 0, 68.53, 67.84, 65.60, 67.57, 69.11, 70.00, 69.41, 65.98, 70.04, 70.13]

#     # model's parameters
#     infer_time = [ 0, 0, 2000, 730, 597, 1000, 2000, 2000, 1000, 470, 832, 701]
#     parameters = [0, 0, 20875248, 3235463, 973135, 2284615, 2284615, 
#                 6671671, 264679, 58423, 290719, 290719]
#     memory = [0, 0, 250.70, 39.00, 12.40, 27.90, 27.90, 80.30, 3.90, 0.898, 3.90, 3.90]


#     fig, (ax, ax2) = plt.subplots(2, 1)
#     ax.set_title('click on point to plot time series')
#     line, = ax.plot(index, accuracy, 'o')  # 5 points tolerance

#     browser = PointBrowser()

#     fig.canvas.mpl_connect('pick_event', browser.onpick)
#     fig.canvas.mpl_connect('key_press_event', browser.onpress)

#     plt.show()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

if 1:  # simple picking, lines, rectangles and text
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('click on points, rectangles or text', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance

    # pick the rectangle
    bars = ax2.bar(range(10), rand(10), picker=True)
    for label in ax2.get_xticklabels():  # make the xtick labels pickable
        label.set_picker(True)

    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick1 patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onpick1 text:', text.get_text())

    fig.canvas.mpl_connect('pick_event', onpick1)



plt.show()