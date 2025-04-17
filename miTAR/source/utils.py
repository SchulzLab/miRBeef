import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
import matplotlib as mpl


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(mpl.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                          + np.array([left_edge, base])[None, :]),
                                         facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                     facecolor=color, edgecolor=color))
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                     facecolor='white', edgecolor='white'))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                       facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                     facecolor=color, edgecolor=color))
    ax.add_patch(mpl.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                     facecolor='white', edgecolor='white'))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                       facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        mpl.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                              facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                       facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge + 0.4, base],
                                       width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(mpl.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                       width=1.0, height=0.2 * height, facecolor=color, edgecolor=color, fill=True))


def plot_weights_given_ax(ax, array,
                          figsize=(20, 2),
                          height_padding_factor=0.2,
                          length_padding=1.0,
                          subticks_frequency=1.0,
                          colors={0: 'green', 1: 'red', 2: 'blue', 3: 'orange'},
                          plot_funcs={0: plot_a, 1: plot_t, 2: plot_c, 3: plot_g},
                          highlight={},
                          ylabel=""):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                mpl.patches.Rectangle(xy=[start_pos, min_depth],
                                      width=end_pos - start_pos,
                                      height=max_height - min_depth,
                                      edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * height_padding_factor,
                         abs(max_pos_height) * height_padding_factor)
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontsize(15)


def plot_weights(array,
                 figsize=(20, 2),
                 despine=False,
                 **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(ax=ax, array=array, **kwargs)
    if despine:
        plt.axis('off')
    # plt.show()
    return fig, ax


def generate_word2vec_model(input_file, window_size=3, feature_size=16, train_ratio=0.81, output_dir="./data/mitar/word2vec-models/"):
    df = pd.read_csv(input_file, sep='\t', header=0)
    df["miRNA_seq"] = df["miRNA_seq"].str.strip("N")
    df["target_seq"] = df["target_seq"].str.strip("N")
    df["mirna_words"] = df["miRNA_seq"].apply(lambda x: [x[i:i+window_size] for i in range(0, len(x), window_size)])
    df["target_words"] = df["target_seq"].apply(lambda x: [x[i:i+window_size] for i in range(0, len(x), window_size)])
    train_df = df.sample(frac=train_ratio, random_state=42)
    sentences_mirna = list(train_df["mirna_words"])
    sentences_target = list(train_df["target_words"])
    model_mirna = Word2Vec(sentences=sentences_mirna, vector_size=feature_size, window=window_size, min_count=1, workers=4)
    model_mirna.save(os.path.join(output_dir, "mirna.model"))
    model_target = Word2Vec(sentences=sentences_target, vector_size=feature_size, window=window_size, min_count=1, workers=4)
    model_target.save(os.path.join(output_dir, "target.model"))