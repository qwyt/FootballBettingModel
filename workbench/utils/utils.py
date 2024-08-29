import pandas as pd


def pandas_config(pd):
    pd.set_option("max_colwidth", 8000)
    pd.options.display.max_rows = 1000
    pd.set_option("display.width", 500)
    pd.set_option("display.max_colwidth", 5000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 2000)

    pd.set_option("display.max_columns", 200)


def plt_config(plt):
    plt.style.use("fivethirtyeight")
