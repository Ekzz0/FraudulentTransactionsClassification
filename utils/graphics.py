import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(6, 4))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def plot_corr_matrix(X):
    h = X[X.columns]

    # whitegrid
    sns.set_style('whitegrid')
    # compute correlation matrix...
    corr_matrix = h.corr(method='spearman')
    # ...and show it with a heatmap
    # first define the dimension
    # plt.figure(figsize=(10, 8))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmax=1, vmin=-1, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


def plot_metrics_hist(models_names, metrics):
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
    col_map = plt.get_cmap('Paired')
    ax.bar(models_names, metrics, color=col_map.colors, edgecolor='k')
    plt.grid(color='r', linestyle='--', linewidth=1)
    ax.set_title("Распределение ROC_AUC_SCORE по моделям")
    ax.set_xlabel("Модель")
    ax.set_ylabel("ROC_AUC_SCORE")
    fig.set_figwidth(10)  # ширина Figure
    plt.ylim(min(metrics)*0.90, max(metrics)*1.03)


def plot_class_separation(X, y, class_names, pairs, fig_size=3):
    from matplotlib.colors import ListedColormap
    cmap_bold = ListedColormap(['#FF0000',  '#00FF00'])

    n_pairs = len(pairs)

    fig, ax = plt.subplots(nrows=n_pairs, ncols=1, figsize=(fig_size, n_pairs*fig_size))

    for j, pair in enumerate(pairs):
        # отрисуем экземпляры
        for i, iris_class in enumerate(class_names):
            idx = y==i
            ax[j].scatter(X[idx][pair[0]], X[idx][pair[1]],
                          c=cmap_bold.colors[i], edgecolor='k',
                          s=20, label=iris_class)

        ax[j].set(xlabel=pair[0], ylabel=pair[1])
        ax[j].legend()
