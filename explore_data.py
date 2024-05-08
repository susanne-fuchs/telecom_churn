import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def explore_values(file_path):
    pd.set_option('display.max_columns', 21)
    df = pd.read_csv(file_path)

    # 1st overview
    print(df.head())
    # # Which columns are numerical
    print(df.dtypes)
    # # Statistics of numerical types
    print(df.describe())
    # # Number of NaNs per column
    print(df.isna().sum())


def statistics(data_path):
    """
    Creates a html file of a statistical overview over numerical columns in a data file.
    """
    df = pd.read_csv(data_path)
    stats_df = df.describe()
    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    html_path = os.path.join(directory, 'stats_' + filename + '.html')
    stats_df.to_html(html_path)


def pair_plot(file_path, plot_path):
    df = pd.read_csv(file_path)
    palette = {"Stayed": 'blue', "Left": 'red'}
    sns.pairplot(df, plot_kws={'alpha': 0.2}, hue='Churn')
    ensure_dir_exists(plot_path)
    plt.savefig(plot_path)


def hist_plot(file_path, plot_path):
    df = pd.read_csv(file_path)
    df = convert_columns(df)

    # Style
    palette = {'Stayed': 'dimgray', 'Left': 'firebrick'}
    sns.set_style("dark")
    # sns.set_style({'axes.facecolor': '#ffffff'})
    #sns.despine(bottom=False, left=True)

    # Number of variables, rows and columns
    var_nmbr = df.shape[1] - 1  # without label
    nrows = 3
    ncols = int(np.ceil(var_nmbr / nrows))

    # Create subplots.
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = axs.flatten()

    # draw kde plot for all variables, each including one plot for customers who churned and who stayed.
    for i, column in enumerate(df.drop(columns=['Churn'])):
        if column in ['Complaints', 'Tariff Plan', 'Status']:
            # Complaints has only two categories
            sns.histplot(data=df, x='Churn', hue=column, multiple='fill', shrink=.55, ax=axs[i])
            # sns.histplot(df, x=column, stat='percent', common_norm=True, multiple="dodge",
            #              shrink=.55,
            #              hue='Churn', palette=palette, ax=axs[i])
            #sns.catplot(data=df, y=column, x='Churn', ax=axs[i])
        else:
            sns.kdeplot(df, x=column, common_norm=False, hue='Churn', hue_order=['Left', 'Stayed'],
                        fill=False, ax=axs[i], palette=palette)
            if i > 0:
                axs[i].get_legend().remove()
        sns.despine(bottom=False, left=True)
        axs[i].set_title(column)
        axs[i].set(ylabel=None, xlabel=None)
        axs[i].tick_params(labelleft=False, left=False, bottom=False, labelbottom=True)

    # Delete empty plots
    for i in range(var_nmbr, nrows * ncols):
        fig.delaxes(axs[i])

    # using padding
    plt.close(2)
    plt.tight_layout()
    plt.savefig(plot_path)


def analyse_usage(file_path, plot_path, categorical_var):
    df = pd.read_csv(file_path)
    df = convert_columns(df)
    column = "Frequency of use"
    fig, axs = plt.subplots()

    # Number of usage related variables, rows and columns
    usage_vars = ["Subscription Length", "Charge Amount", "Minutes of Use", "Frequency of use", "Frequency of SMS",
                  "Distinct Called Numbers"]
    var_nmbr = len(usage_vars)
    nrows = 2
    ncols = int(np.ceil(var_nmbr / nrows))

    # Create subplots.
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = axs.flatten()

    # draw kde plot for variables, each including one plot for customers who churned and who stayed.
    for i, column in enumerate(df[usage_vars]):
        sns.kdeplot(df, x=column, hue=categorical_var, common_norm=False, fill=False, ax=axs[i]) #, palette=palette)
        if i > 0:
            axs[i].get_legend().remove()
        sns.despine(bottom=False, left=True)
        axs[i].set_title(column)
        axs[i].set(ylabel=None, xlabel=None)
        axs[i].tick_params(labelleft=False, left=False, bottom=False, labelbottom=True)

    plt.tight_layout()
    plt.savefig(plot_path)


def convert_columns(df):
    """
    Converts columns with only two values to categorical data and converts time from seconds to minutes.
    """
    # Convert seconds to minutes:
    df["Seconds of Use"] = df["Seconds of Use"] / 60
    df.rename(columns={"Seconds of Use": "Minutes of Use"}, inplace=True)

    # Convert Complaints to yes/no
    df['Complaints'] = df['Complaints'].map({0: 'No', 1: 'Yes'})
    df['Complaints'] = pd.Categorical(df['Complaints'], ordered=True, categories=['No', 'Yes'])

    # Convert Tariff Plan to categorical
    for col in ['Tariff Plan', 'Status']:
        df[col] = df[col].map({1: '1', 2: '2'})
        df[col] = pd.Categorical(df[col], ordered=True, categories=['1', '2'])

    # Convert churn to yes/no
    # df['Churn'] = df['Churn'].map({0: False, 1: True})
    df['Churn'] = df['Churn'].map({0: 'Stayed', 1: 'Left'})
    df['Churn'] = pd.Categorical(df['Churn'], ordered=True, categories=['Stayed', 'Left'])
    return df


def ensure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def print_crosstab(file_path):
    for col in ['Status', 'Tariff Plan', 'Complaints']:
        for normalize in [False, 'index', 'columns']:
            save_path = f'Results/crosstab_{col.lower().replace(" ", "_")}_churn_{str(normalize)}.svg',
            ct = crosstab(file_path, save_path,  col, 'Churn', normalize)
            print(ct)


def crosstab(file_path, save_path, var0, var1, normalize):
    df = pd.read_csv(file_path)
    df = convert_columns(df)

    ct = pd.crosstab(df[var0], df[var1], normalize=normalize)
    return ct


def explore_data():
    file_path = 'Data/customer_churn.csv'
    explore_values(file_path)

    pair_plot(file_path,
              'Results/pairplot.png')
