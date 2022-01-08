import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')


def plot_top3_amt(df, plot_type='amt'):
    xlabels = df['dt'].to_list()
    if plot_type == 'amt':
        ylabels_top1 = df['top1_amt'].to_list()
        ylabels_top2 = df['top2_amt'].to_list()
        ylabels_top3 = df['top3_amt'].to_list()
    elif plot_type == 'cnt':
        ylabels_top1 = df['top1_cnt'].to_list()
        ylabels_top2 = df['top2_cnt'].to_list()
        ylabels_top3 = df['top3_cnt'].to_list()
    width = 0.2
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(np.array(xlabels) - width, ylabels_top1, width, label='top1')
    ax.bar(np.array(xlabels),
           ylabels_top2,
           width,
           color='orange',
           label='top2')
    ax.bar(np.array(xlabels) + width,
           ylabels_top3,
           width,
           color='green',
           label='top3')

    top1 = df['top1'].to_list()
    top2 = df['top2'].to_list()
    top3 = df['top3'].to_list()
    if plot_type == 'amt':
        ax.set_title("Amt for each dt")
    elif plot_type == 'cnt':
        ax.set_title("Cnt for each dt")
    for i, v in enumerate(ylabels_top1):
        ax.text(xlabels[i] - .25,
                v + 0.1,
                top1[i],
                color='blue',
                fontweight='bold')

    for i, v in enumerate(ylabels_top2):
        ax.text(xlabels[i], v + 0.1, top2[i], color='blue', fontweight='bold')

    for i, v in enumerate(ylabels_top3):
        ax.text(xlabels[i] + .25,
                v + 0.1,
                top3[i],
                color='blue',
                fontweight='bold')

    ax.legend(loc='upper right', shadow=True)

    return fig


def plot_top3_cnt(df):
    pass
