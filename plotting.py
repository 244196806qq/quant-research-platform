import matplotlib.pyplot as plt


def graph(dfs, df_strategy, symbols, strategy, ax2):
    for i in range(len(dfs)):
        ax2.plot(dfs[i].index, dfs[i]["Market_Equity"], label = f"Buy and Hold {symbols[i]}")
    for i in range(len(df_strategy.columns)):
        ax2.plot(df_strategy.index, df_strategy[strategy[i]], label = f"Strategy {strategy[i]}")
    ax2.legend()
    ax2.grid(True)

def open_window(dfs, df_strategy, symbols, strategy, final_money):
    data, columns = show_return(dfs, df_strategy, symbols, strategy, final_money)

    fig, (ax1, ax2) = plt.subplots(
        nrows = 1, 
        ncols = 2, 
        figsize = (14,10), 
        gridspec_kw = {"width_ratios": [1,3]}
    )
    ax1.axis("off")
    graph(dfs, df_strategy, symbols, strategy, ax2)
    table = ax1.table(
        cellText = data,
        colLabels = columns,
        loc = "center"
    )
    table.scale(1,2)
    plt.show()

def show_return(dfs, df_strategy, symbols, strategy, final_money):
    data = []
    for i in range(len(dfs)):
        data.append([f"Buy and Hold {symbols[i]}", f"{calculate_return(dfs[i]["Market_Equity"], final_money):.2f}%"])
    for i in range(len(df_strategy.columns)):
        data.append([f"Strategy {strategy[i]}", f"{calculate_return(df_strategy[strategy[i]], final_money):.2f}%"])

    columns = ["Strategy", "Return"]
    return data, columns

def calculate_return(df_asset, final_money):
    percent_inc = 100 * (df_asset.iloc[-1] - final_money) / final_money
    return percent_inc
        
