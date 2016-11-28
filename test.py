import numpy as np
import matplotlib.pyplot as plt

prob_metrices, prob_labels = np.array(), np.array()
img_list = np.array()

N = len(prob_labels)

top = prob_metrices[:, 0]

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, top, width, color='r')

top_2 = prob_metrices[:, 1]
rects2 = ax.bar(ind + width, top_2, width, color='y')

top_3 = prob_metrices[:, 2]
rects3 = ax.bar(ind + width, top_3, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('Probabilty')
ax.set_title('Scores by image and top 3')
ax.set_xticks(ind + 2*width)
ax.set_xticklabels([pic_name for pic_name in img_list])

ax.legend((rects1[0], rects2[0], rects3[0]), ('1', '2', '3'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()