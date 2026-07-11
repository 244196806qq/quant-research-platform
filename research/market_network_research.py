import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import glob
import pickle
import time
import json
from fa2_modified import ForceAtlas2
from datetime import date, timedelta
from sklearn.feature_selection import mutual_info_regression
from networkx.algorithms.community import greedy_modularity_communities
from pathlib import Path

ROLLING_DAYS = 300
TODAY_DATE = date.today()
TODAY_WEEKDAY = TODAY_DATE.weekday()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_FILE = DATA_DIR / "stock_metadata.json"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DF_STOCKS_CACHE = CACHE_DIR / "df_stocks.pkl"
SYMBOLS_CACHE = CACHE_DIR / "symbols.pkl"
RETURNS_CACHE = CACHE_DIR / f"returns_{ROLLING_DAYS}.pkl"
EMPIRICAL_MI_CACHE = CACHE_DIR / f"empirical_mi_{ROLLING_DAYS}.pkl"
PEARSON_CORR_CACHE = CACHE_DIR / f"pearson_corr_{ROLLING_DAYS}.pkl"
GEOM_DIST_CACHE = CACHE_DIR / f"pearson_dist_{ROLLING_DAYS}.pkl"
MI_DIST_CACHE = CACHE_DIR / f"mi_dist_{ROLLING_DAYS}.pkl"
ROLLING_FREQ_CACHE = CACHE_DIR / f"rolling_freq_semiconductors_{ROLLING_DAYS}_21_top10.pkl"


# Helper
def load_stock_metadata():
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def has_enough_history(df: pd.DataFrame, min_trading_days: int = ROLLING_DAYS) -> bool:
    if "Adj Close" not in df.columns:
        return False

    valid_prices = df["Adj Close"].dropna()

    return len(valid_prices) >= min_trading_days + 1

def get_tickers_in_same_group(metadata, tickers, target_ticker, level):
    if target_ticker not in tickers:
        return []
    target_info = metadata[target_ticker]
    target_level = target_info.get(level)
    if not target_level:
        return []
    
    same_group = []
    for ticker in tickers:
        if ticker == target_ticker:
            continue

        info = metadata.get(ticker, {})
        if info.get(level) == target_level:
            same_group.append(ticker)
    return same_group


# Cache
def cache_is_fresh(stock_files):
    if not DF_STOCKS_CACHE.exists() or not SYMBOLS_CACHE.exists():
        return False
    cache_time = DF_STOCKS_CACHE.stat().st_mtime
    for file in stock_files:
        if file.stat().st_mtime > cache_time:
            return False
        
    return True

def delete_cache_if_older(cache_file: Path, dependency_file: Path):
    if(cache_file.exists() and dependency_file.exists() and dependency_file.stat().st_mtime > cache_file.stat().st_mtime):
        print(f"Deleting outdated cache: {cache_file}")
        cache_file.unlink()

def get_all_stock_files():
    stock_files = []
    for stock_folder in DATA_DIR.rglob("_stocks"):
        stock_files.extend(stock_folder.glob("*.csv"))
    return sorted(stock_files)

def load_stock_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def load_or_compute(cache_file, compute_function):
    if cache_file != ROLLING_FREQ_CACHE:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        result = compute_function()
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    else:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                result = pickle.load(f)

            return result["detail_table"], result["frequency_table"]
        
        detail_table, frequency_table = compute_function()
        result = {
            "detail_table": detail_table,
            "frequency_table": frequency_table,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    
    return result

def load_or_cache_df_stocks(force_reload: bool = False, min_trading_days: int = ROLLING_DAYS):
    stock_files = get_all_stock_files()

    if not force_reload and cache_is_fresh(stock_files):
        with open(DF_STOCKS_CACHE, "rb") as f:
            df_stocks = pickle.load(f)

        with open(SYMBOLS_CACHE, "rb") as f:
            symbols = pickle.load(f)

        return symbols, df_stocks

    symbols = []
    df_stocks = []
    skipped = []

    for file in stock_files:
        symbol = file.stem
        df = load_stock_csv(file)

        if not has_enough_history(df, min_trading_days=min_trading_days):
            skipped.append(symbol)
            continue

        symbols.append(symbol)
        df_stocks.append(df)

    with open(DF_STOCKS_CACHE, "wb") as f:
        pickle.dump(df_stocks, f)

    with open(SYMBOLS_CACHE, "wb") as f:
        pickle.dump(symbols, f)

    if skipped:
        print(f"Skipped {len(skipped)} stocks with less than {min_trading_days} trading days:")
        print(skipped)

    return symbols, df_stocks

# Correlation Calculation
def price_to_return(df) -> float:
     return np.log(df["Adj Close"] / df["Adj Close"].shift(1))

def pearson_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    # Equation: cov = (1/(len(df1)-1)) * ((df1["Market_Return"] - df1["Market_Return"].mean()) * (df2["Market_Return"] - df2["Market_Return"].mean())).sum()
    n = returns.shape[1]
    pearson_corr = pd.DataFrame(np.zeros((n,n)), index=returns.columns, columns=returns.columns)
    
    for i in range(n):
        for j in range(i, n):
            df = pd.concat([returns.iloc[:, i], returns.iloc[:, j]], axis=1).dropna()
            cov = df.iloc[:,0].cov(df.iloc[:,1])
            denom = df.iloc[:,0].std() * df.iloc[:,1].std()
            if denom == 0 or pd.isna(denom):
                corr = np.nan
            else:
                corr = cov / denom
            pearson_corr.iloc[i, j] = corr
            pearson_corr.iloc[j, i] = corr
    
    return pearson_corr

def pearson_distance(pearson_corr: pd.DataFrame) -> pd.DataFrame:
    corr = pearson_corr.clip(lower=-1, upper=1)
    pearson_dist = np.sqrt(2 * (1 - corr))
    np.fill_diagonal(pearson_dist.values, 0.0)
    return pearson_dist

def mi_distance(empirical_mi_df: pd.DataFrame) -> pd.DataFrame:
    mi_similarity = np.sqrt(1 - np.exp(-2 * empirical_mi_df))

    mi_dist = 1 - mi_similarity

    np.fill_diagonal(mi_dist.values, 0.0)

    return mi_dist

def compute_gaussian_mi_matrix(pearson_corr: pd.DataFrame) -> pd.DataFrame:
    gaussian_mi = -0.5 * np.log(1 - pearson_corr ** 2)
    np.fill_diagonal(gaussian_mi.values, np.nan)

    return gaussian_mi

def compute_nonlinear_info(pearson_corr: pd.DataFrame, empirical_mi_df: pd.DataFrame) -> pd.DataFrame:
    gaussian_mi = compute_gaussian_mi_matrix(pearson_corr)
    nonlinear_mi = empirical_mi_df - gaussian_mi 
    return nonlinear_mi

def compute_MI_matrix_df(returns: pd.DataFrame, min_overlap: int = ROLLING_DAYS,) -> pd.DataFrame:
    n = returns.shape[1]
    empirical_mi_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            df = pd.concat([returns.iloc[:, i], returns.iloc[:, j]], axis=1).dropna()

            if len(df) < min_overlap:
                continue
            if df.iloc[:, 0].std() == 0 or df.iloc[:, 1].std() == 0:
                continue

            mi = mutual_info_regression(X=df.iloc[:, [0]], y=df.iloc[:, 1], random_state=42)[0]
            empirical_mi_matrix[i, j] = mi
            empirical_mi_matrix[j, i] = mi

    return pd.DataFrame(empirical_mi_matrix, index=returns.columns, columns=returns.columns)

def average_distance_between_groups(distance_df, tickers_a, tickers_b):
    distances = []
    for a in tickers_a:
        for b in tickers_b:
            if a == b:
                continue
            if a not in distance_df.columns or b not in distance_df.columns:
                continue
            dist = distance_df.loc[a, b]
            if not pd.isna(dist):
                distances.append(dist)

    if not distances:
        return np.nan

    return float(np.mean(distances))

def mean_mi_for_stock(empirical_mi_df: pd.DataFrame, ticker: str) -> float:
    if ticker not in empirical_mi_df.columns:
        return np.nan

    values = (
        empirical_mi_df[ticker]
        .drop(index=ticker, errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(values) == 0:
        return np.nan

    return float(values.mean())

def mean_mi_against_tickers(empirical_mi_df, ticker, other_tickers):
    if ticker not in empirical_mi_df.columns:
        return np.nan
    
    valid_tickers = [
        t for t in other_tickers if t in empirical_mi_df.columns and t != ticker
    ]
    if len(valid_tickers) == 0:
        return np.nan

    values = (
        empirical_mi_df.loc[valid_tickers, ticker].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(values) == 0:
        return np.nan
    
    return float(values.mean())

def normalized_price_series(prices_df: pd.DataFrame, tickers: list, lookback: int = 63):
    prices = prices_df[tickers].dropna(how="all").tail(lookback)
    prices = prices.dropna(axis=1)

    return prices / prices.iloc[0]


# Build Network
def build_network_from_distance(dist_df: pd.DataFrame):
    G = nx.Graph()

    for ticker in dist_df.columns:
        G.add_node(ticker)

    n = len(dist_df.columns)
    for i in range(n):
        for j in range(i+1, n):
            a = dist_df.columns[i]
            b = dist_df.columns[j]
            dist = dist_df.loc[a, b]
            if pd.isna(dist):
                continue
            G.add_edge(
                a,
                b,
                weight = dist,
            )
    return G

def build_pmfg(returns: pd.DataFrame, pearson_dist: pd.DataFrame):
    stocks = list(returns.columns)
    n = len(stocks)

    edges = []
    for i in range(n):
        u = stocks[i]
        for j in range(i+1, n):
            v = stocks[j]
            dist = pearson_dist.iloc[i,j]
            similarity = 0.0
            if pd.isna(dist):
                continue
            if dist == 0.0:
                similarity = np.inf
            else:
                similarity = 1 / dist
            edges.append((u,v,similarity,dist))

    edges.sort(key=lambda x: x[2], reverse=True)

    pmfg = nx.Graph()
    for stock in stocks:
        pmfg.add_node(stock)
    
    max_edges = 3 * n - 6
    for u,v,similarity, dist in edges:
        pmfg.add_edge(
            u, v, weight=dist, similarity=similarity,
        )
        is_planar, _ = nx.check_planarity(pmfg)
        if not is_planar:
            pmfg.remove_edge(u,v)
        if pmfg.number_of_edges() >= max_edges:
            break
    
    return pmfg

def group_tickers_by_sector_industry(tickers, metadata):
    groups = {}
    for ticker in tickers:
        info = metadata.get(ticker, {})
        sector = info.get("sector")
        industry = info.get("industry")
        if not sector or not industry:
            continue

        groups.setdefault(sector, {})
        groups[sector].setdefault(industry, [])
        groups[sector][industry].append(ticker)

    return groups

def build_mst_for_tickers(distance_df, tickers):
    tickers = [ticker for ticker in tickers if ticker in distance_df.columns]
    if len(tickers) < 2:
        return None
    
    sub_dist = distance_df.loc[tickers, tickers]
    G = build_network_from_distance(sub_dist)
    mst = nx.minimum_spanning_tree(G, weight="weight")

    return mst

def build_industry_msts_by_sector(dist_df: pd.DataFrame, metadata):
    grouped = group_tickers_by_sector_industry(dist_df.columns, metadata)

    sector_industry_msts = {}
    for sector, industries in grouped.items():
        sector_industry_msts[sector] = {}
        for industry, tickers in industries.items():
            mst = build_mst_for_tickers(dist_df, tickers)
            if mst is not None:
                sector_industry_msts[sector][industry] = mst

    return sector_industry_msts

def build_stock_graph_for_tickers(distance_df, tickers):
    tickers = [t for t in tickers if t in distance_df.columns]

    G = nx.Graph()

    for ticker in tickers:
        G.add_node(ticker)

    if len(tickers) < 2:
        return G

    sub_dist = distance_df.loc[tickers, tickers]

    full_graph = build_network_from_distance(sub_dist)
    mst = nx.minimum_spanning_tree(full_graph, weight="weight")

    return mst


# Plotting Network
def categorize(returns: pd.DataFrame, labels: list):
    sigma = returns.std()
    bins = [-np.inf, -2*sigma, -sigma, -0.002, 0.002, sigma, 2*sigma, np.inf]
    return pd.cut(returns, bins=bins, labels=labels)

def plot_pmfg(returns: pd.DataFrame, pearson_dist: pd.DataFrame):
    pmfg = build_pmfg(returns, pearson_dist)

    plt.figure(figsize=(10,8))

    # forceatlas2 = ForceAtlas2()
    mst = nx.minimum_spanning_tree(pmfg)
    pos = nx.spring_layout(mst, seed=42)
    weights = nx.get_edge_attributes(pmfg, "weight")
    widths = [0.5 for _ in weights.values()]

    # nx.draw_networkx_edges(
    #     pmfg, pos, width=widths, alpha=0.15,
    # )
    # nx.draw_networkx_edges(
    #     pmfg,
    #     pos,
    #     alpha=0.1,
    #     width=0.5
    # )

    nx.draw_networkx_edges(
        mst,
        pos,
        width=3,
        edge_color='red'
    )
    nx.draw_networkx_nodes(
        pmfg, pos, node_size=700,
    )
    nx.draw_networkx_labels(
        pmfg, pos, font_size=8
    )
    plt.title("PMFG Market Movement Network")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_min_spanning_tree(pearson_dist: pd.DataFrame, title: str="Who tends to move together in the market?"):
    G = build_network_from_distance(pearson_dist)
    mst = nx.minimum_spanning_tree(G)

    communities = greedy_modularity_communities(mst, weight="weight")

    pos = {}
    for k, community in enumerate(communities):
        subgraph = mst.subgraph(community)
        sub_pos = nx.spring_layout(subgraph, seed=42, k=1.2)

        offset_x = k * 4
        for node in subgraph.nodes:
            pos[node] = sub_pos[node] + np.array([offset_x, 0])

    plt.figure(figsize=(10, 8))

    nx.draw_networkx_edges(mst, pos, width=2)

    nx.draw_networkx_nodes(
        mst,
        pos,
        node_size=700
    )

    nx.draw_networkx_labels(
        mst,
        pos,
        font_size=8
    )

    plt.axis("off")
    plt.tight_layout()
    plt.title(title)
    plt.show()

def plot_states_correlation_heatmap(returns: pd.DataFrame, title: str = "Joint Probability"):
    labels = [
        "Strong Drop",
        "Medium Drop",
        "Weak Drop",
        "Flat",
        "Weak Rise",
        "Medium Rise",
        "Strong Rise"
    ]
    states = returns.apply(categorize, labels = labels)
    n = len(labels)

    mi_matrix = pd.DataFrame(
        np.zeros((n, n)),
        index=labels,
        columns=labels
    )
    for i, state_i in enumerate(labels):
        for j, state_j in enumerate(labels):
            count = ((states.iloc[:, 0] == state_i) & (states.iloc[:, 1] == state_j)).sum()
            mi_matrix.iloc[i, j] = count / len(states)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mi_matrix, cmap="viridis")

    plt.xticks(range(n), labels, rotation=0)
    plt.yticks(range(n), labels)

    stock_x = states.columns[0]
    stock_y = states.columns[1]
    plt.xlabel(f"{stock_x} States")
    plt.ylabel(f"{stock_y} States")

    plt.colorbar(label="Joint Probability")
    plt.title("title")

    plt.show()

def build_industry_msts_by_sector(distance_df, metadata):
    groups = group_tickers_by_sector_industry(distance_df.columns, metadata)
    result = {}
    for sector, industries in groups.items():
        result[sector] = {}
        for industry, tickers in industries.items():
            mst = build_mst_for_tickers(distance_df, tickers)
            if mst is not None:
                result[sector][industry] = mst

    return result

def build_industry_connection_network(distance_df, metadata, sector):
    groups = group_tickers_by_sector_industry(distance_df.columns, metadata)
    if sector not in groups:
        raise ValueError(f"Sector not found: {sector}")

    industries = groups[sector]
    G = nx.Graph()
    for industry in industries:
        G.add_node(industry)

    industry_names = list(industries.keys())
    for i in range(len(industry_names)):
        for j in range(i + 1, len(industry_names)):
            industry_a = industry_names[i]
            industry_b = industry_names[j]
            dist = average_distance_between_groups(
                distance_df,
                industries[industry_a],
                industries[industry_b],
            )
            if pd.isna(dist):
                continue

            G.add_edge(
                industry_a,
                industry_b,
                weight=dist,
                similarity=1 / dist if dist != 0 else np.inf,
            )

    return G

def build_sector_industry_msts(sector: str, industry_msts: dict):
    plt.figure(figsize=(10, 10))
    combined_graph = nx.Graph()
    pos = {}
    x_offset = 0

    for industry, mst in industry_msts.items():
        sub_pos = nx.spring_layout(mst, seed=42)
        for node, xy in sub_pos.items():
            pos[node] = xy + np.array([x_offset, 0])
        combined_graph = nx.compose(combined_graph, mst)
        plt.text(
            x_offset,
            1.4,
            industry,
            fontsize=12,
            ha="center",
            fontweight="bold",
        )
        x_offset += 4

    nx.draw_networkx_edges(combined_graph, pos, width=2)
    nx.draw_networkx_nodes(combined_graph, pos, node_size=700)
    nx.draw_networkx_labels(combined_graph, pos, font_size=8)

    plt.title(f"{sector} Industry MSTs")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def closest_stock_pair_between_groups(distance_df, tickers_a, tickers_b):
    best_pair = None
    best_dist = np.inf

    for a in tickers_a:
        for b in tickers_b:
            if a == b:
                continue

            if a not in distance_df.columns or b not in distance_df.columns:
                continue

            dist = distance_df.loc[a, b]

            if pd.isna(dist):
                continue

            if dist < best_dist:
                best_dist = dist
                best_pair = (a, b)

    return best_pair, best_dist

def plot_industry_connection_mst(distance_df, metadata, sector):
    G = build_industry_connection_network(distance_df, metadata, sector)
    mst = nx.minimum_spanning_tree(G, weight="weight")
    pos = nx.spring_layout(mst, seed=42)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(mst, pos, width=2)
    nx.draw_networkx_nodes(mst, pos, node_size=1200)
    nx.draw_networkx_labels(mst, pos, font_size=10)

    edge_labels = {
        (u, v): f"{d['weight']:.3f}"
        for u, v, d in mst.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        mst,
        pos,
        edge_labels=edge_labels,
        font_size=8,
    )

    plt.title(f"{sector} Industry Connection MST")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_full_sector_network(distance_df, metadata, sector):
    groups = group_tickers_by_sector_industry(distance_df.columns, metadata)

    if sector not in groups:
        raise ValueError(f"Sector not found: {sector}")

    industries = groups[sector]

    # Industry-level MST tells us which industries should be connected.
    industry_connection_graph = build_industry_connection_network(
        distance_df,
        metadata,
        sector,
    )

    industry_mst = nx.minimum_spanning_tree(
        industry_connection_graph,
        weight="weight",
    )

    combined_graph = nx.Graph()
    pos = {}
    industry_titles = {}

    # Control spacing here.
    x_gap = 7
    y_gap = 9
    industries_per_row = 3

    industry_order = list(industries.keys())

    for idx, industry in enumerate(industry_order):
        tickers = industries[industry]

        mst = build_stock_graph_for_tickers(distance_df, tickers)

        if mst.number_of_nodes() == 0:
            continue

        row = idx // industries_per_row
        col = idx % industries_per_row

        center = np.array([
            col * x_gap,
            -row * y_gap,
        ])

        # Use a larger k and more iterations to reduce overlap.
        if mst.number_of_nodes() == 1:
            sub_pos = {
                list(mst.nodes())[0]: np.array([0.0, 0.0])
            }
        else:
            sub_pos = nx.spring_layout(
                mst,
                seed=42,
                k=3.5,
                iterations=500,
            )

        layout_scale = 4.0
        # Scale each cluster outward.
        for node, xy in sub_pos.items():
            pos[node] = (xy * layout_scale) + center

        combined_graph = nx.compose(combined_graph, mst)

        # Industry title, not a node.
        industry_titles[industry] = center + np.array([0, 3.2])

    plt.figure(figsize=(8, 8))

    # Draw stock MST edges inside industries.
    nx.draw_networkx_edges(
        combined_graph,
        pos,
        width=1.4,
        alpha=0.75,
    )

    nx.draw_networkx_nodes(
        combined_graph,
        pos,
        node_size=520,
    )

    nx.draw_networkx_labels(
        combined_graph,
        pos,
        font_size=7,
    )

    # Draw industry titles.
    for industry, title_pos in industry_titles.items():
        plt.text(
            title_pos[0],
            title_pos[1],
            industry,
            fontsize=13,
            ha="center",
            fontweight="bold",
        )

    # Draw dashed industry connections, but between real stock nodes.
    for industry_a, industry_b, data in industry_mst.edges(data=True):
        tickers_a = industries[industry_a]
        tickers_b = industries[industry_b]

        pair, dist = closest_stock_pair_between_groups(
            distance_df,
            tickers_a,
            tickers_b,
        )

        if pair is None:
            continue

        stock_a, stock_b = pair

        if stock_a not in pos or stock_b not in pos:
            continue

        x1, y1 = pos[stock_a]
        x2, y2 = pos[stock_b]

        plt.plot(
            [x1, x2],
            [y1, y2],
            linestyle="--",
            linewidth=1.8,
            alpha=0.55,
        )

    plt.title(f"{sector} Industry Stock MSTs")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def build_mean_mi_table_robust(empirical_mi_df: pd.DataFrame, overlap_count_df: pd.DataFrame, metadata: dict, min_overlap: int = 252,) -> pd.DataFrame:
    tickers = list(empirical_mi_df.columns)
    rows = []

    for ticker in tickers:
        info = metadata.get(ticker, {})

        sector = info.get("sector")
        industry = info.get("industry")
        subindustry = info.get("subindustry")

        same_sector = [
            t for t in tickers
            if metadata.get(t, {}).get("sector") == sector
            and t != ticker
        ]

        same_industry = [
            t for t in tickers
            if metadata.get(t, {}).get("sector") == sector
            and metadata.get(t, {}).get("industry") == industry
            and t != ticker
        ]

        same_subindustry = [
            t for t in tickers
            if subindustry
            and metadata.get(t, {}).get("sector") == sector
            and metadata.get(t, {}).get("industry") == industry
            and metadata.get(t, {}).get("subindustry") == subindustry
            and t != ticker
        ]

        same_sector_diff_industry = [
            t for t in same_sector
            if metadata.get(t, {}).get("industry") != industry
        ]

        overall_stats = mi_stats_against_tickers(
            empirical_mi_df,
            overlap_count_df,
            ticker,
            [t for t in tickers if t != ticker],
            min_overlap=min_overlap,
        )

        industry_stats = mi_stats_against_tickers(
            empirical_mi_df,
            overlap_count_df,
            ticker,
            same_industry,
            min_overlap=min_overlap,
        )

        sector_ex_industry_stats = mi_stats_against_tickers(
            empirical_mi_df,
            overlap_count_df,
            ticker,
            same_sector_diff_industry,
            min_overlap=min_overlap,
        )

        subindustry_stats = mi_stats_against_tickers(
            empirical_mi_df,
            overlap_count_df,
            ticker,
            same_subindustry,
            min_overlap=min_overlap,
        )

        rows.append({
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "subindustry": subindustry,

            "overall_mean_mi": overall_stats["mean_mi"],
            "overall_median_mi": overall_stats["median_mi"],
            # "overall_trimmed_mean_mi": overall_stats["trimmed_mean_mi"],
            # "overall_weighted_mean_mi": overall_stats["weighted_mean_mi"],
            "overall_valid_pairs": overall_stats["num_valid_pairs"],
            "overall_avg_overlap": overall_stats["avg_overlap_days"],

            "industry_mean_mi": industry_stats["mean_mi"],
            "industry_median_mi": industry_stats["median_mi"],
            # "industry_trimmed_mean_mi": industry_stats["trimmed_mean_mi"],
            # "industry_weighted_mean_mi": industry_stats["weighted_mean_mi"],
            "industry_valid_pairs": industry_stats["num_valid_pairs"],
            "industry_avg_overlap": industry_stats["avg_overlap_days"],

            "subindustry_mean_mi": subindustry_stats["mean_mi"],
            "subindustry_valid_pairs": subindustry_stats["num_valid_pairs"],

            "sector_ex_industry_mean_mi": sector_ex_industry_stats["mean_mi"],
            "sector_ex_industry_median_mi": sector_ex_industry_stats["median_mi"],
            "sector_ex_industry_valid_pairs": sector_ex_industry_stats["num_valid_pairs"],
        })

    return pd.DataFrame(rows)

def classify_mi_driver(row, threshold=1.10):
    industry_mi = row["industry_mean_mi"]
    sector_ex_industry_mi = row["sector_ex_industry_mean_mi"]

    if pd.isna(industry_mi) or pd.isna(sector_ex_industry_mi):
        return "unknown"
    
    if industry_mi > sector_ex_industry_mi * threshold:
        return "industry_driven"

    if sector_ex_industry_mi > industry_mi * threshold:
        return "sector_driven"

    return "mixed"

def top_mean_mi_by_group(mean_mi_table, sector=None, industry=None, top_n=3):
    df = mean_mi_table.copy()

    if sector is not None:
        df = df[df["sector"] == sector]
    if industry is not None:
        df = df[df["industry"] == industry]

    return df.sort_values("industry_mean_mi", ascending = False).head(top_n)

def mi_stats_against_tickers(empirical_mi_df: pd.DataFrame, overlap_count_df: pd.DataFrame, ticker: str, overlap_tickers: list, min_overlap: int = 252) -> dict:
    valid_rows = []

    for other in overlap_tickers:
        if other == ticker:
            continue
        if ticker not in empirical_mi_df.columns or other not in empirical_mi_df.index:
            continue

        mi = empirical_mi_df.loc[other, ticker]
        overlap = overlap_count_df.loc[other, ticker]

        if pd.isna(mi):
            continue
        if overlap < min_overlap:
            continue

        valid_rows.append((other, mi, overlap))

    if not valid_rows:
        return {
            "mean_mi": np.nan,
            "median_mi": np.nan,
            "num_valid_pairs": 0,
            "avg_overlap_days": np.nan,
        }

    mi_values = np.array([row[1] for row in valid_rows], dtype=float)
    mi_values = mi_values[~pd.isna(mi_values)]
    overlap_values = np.array([row[2] for row in valid_rows], dtype=float)
    
    if len(mi_values) == 0:
        return {
            "mean_mi": np.nan,
            "median_mi": np.nan,
            "num_valid_pairs": 0,
            "avg_overlap_days": np.nan,
        }

    return {
        "mean_mi": float(np.mean(mi_values)),
        "median_mi": float(np.median(mi_values)),
        "num_valid_pairs": len(valid_rows),
        "avg_overlap_days": float(np.mean(overlap_values)),
    }


# Rolling Frequency
def rolling_top_mi_frequency(returns: pd.DataFrame, metadata: dict, industry: str = "semiconductors", window_length: int = 300, step: int = 21,top_n: int = 20, min_overlap: int = 252, back_years: int=2):
    industry_tickers = [
        ticker for ticker in returns.columns
        if metadata.get(ticker, {}).get("sector") == "technology"
        and metadata.get(ticker, {}).get("industry") == industry
    ]
    industry_returns = returns[industry_tickers].dropna(how="all")
    frequency_dict = {}
    records = []
    eligible_dict = {}
    frequency_score = {}
    n = industry_returns.shape[0]

    for i in range(window_length, n, step):
        rolling_returns = industry_returns.iloc[i-window_length:i]
        window_end_date = rolling_returns.index.max()
        analysis_start_date = industry_returns.index.max() - pd.DateOffset(years=back_years)
        if window_end_date < analysis_start_date:
            continue

        empirical_mi_df = compute_MI_matrix_df(rolling_returns, min_overlap=min_overlap)
        overlap_matrix = compute_overlap_count_matrix(rolling_returns)
        mean_mi_table = build_mean_mi_table_robust(empirical_mi_df, overlap_matrix, metadata, min_overlap=min_overlap)
        industry_table = mean_mi_table[mean_mi_table["industry"] == industry].copy()
        eligible_table = industry_table[industry_table["industry_median_mi"].notna() & (industry_table["industry_valid_pairs"] >= 10)]
        for ticker in eligible_table["ticker"]:
            eligible_dict[ticker] = eligible_dict.get(ticker, 0) + 1

        ranked = (
            eligible_table[
                [
                    "ticker",
                    "industry_mean_mi",
                    "industry_median_mi",
                    "industry_valid_pairs",
                    "industry_avg_overlap",
                ]
            ]
            .sort_values("industry_median_mi", ascending=False).head(top_n)
        )

        decay = 0.90
        for ticker in frequency_score:
            frequency_score[ticker] *= decay
        for ticker in ranked["ticker"]:
            frequency_dict[ticker] = frequency_dict.get(ticker, 0) + 1
            frequency_score[ticker] = frequency_score.get(ticker, 0) + 1

        high_mi_tickers = ranked["ticker"].tolist()
        high_mi_return_divergence = compare_high_mi_return_divergence(rolling_returns, high_mi_tickers, frequency_score, ranked, lookback=63)
        high_mi_price_divergence = compare_high_mi_price_divergence(rolling_returns, high_mi_tickers,frequency_score, ranked, lookback=63)
        high_mi_divergence = high_mi_return_divergence[["ticker", "return_z_score", "latest_residual"]].merge(
            high_mi_price_divergence[["ticker", "price_z_score", "relative_gap"]],
            on="ticker",
            how="inner",
        )
        
        divergence_lookup = high_mi_divergence.set_index("ticker")
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            ticker = row.ticker
            if ticker in divergence_lookup.index:
                divergence = divergence_lookup.loc[ticker]
                return_z = divergence["return_z_score"]
                price_z = divergence["price_z_score"]
                buy_signal = pd.notna(return_z) and pd.notna(price_z) and return_z < -2 or price_z < -2
            else:
                return_z = np.nan
                price_z = np.nan
                buy_signal = False
            records.append({
                "window_end": window_end_date,
                "ticker": ticker,
                "rank": rank,
                "industry_median_mi": row.industry_median_mi,
                "industry_mean_mi": row.industry_mean_mi,
                "industry_valid_pairs": row.industry_valid_pairs,
                "industry_avg_overlap": row.industry_avg_overlap,
                "frequency_score": frequency_score[ticker],
                "return_z_score": return_z,
                "price_z_score": price_z,
                "buy_signal": buy_signal,
            })

    detail_table = pd.DataFrame(records)
    total_windows = detail_table["window_end"].nunique()

    frequency_table = pd.DataFrame(
        list(frequency_dict.items()),
        columns=["ticker", "top_count"]
    )

    frequency_table["eligible_windows"] = frequency_table["ticker"].map(eligible_dict)
    frequency_table["global_frequency"] = frequency_table["top_count"] / total_windows
    frequency_table["conditional_frequency"] = (
        frequency_table["top_count"] / frequency_table["eligible_windows"]
    )
    latest_date = detail_table["window_end"].max()
    latest = detail_table[detail_table["window_end"] == latest_date][
                            ["ticker", "rank", "industry_median_mi"]
                        ].rename(columns=
                                {"rank": "latest_rank","industry_median_mi": "latest_median_mi"}
                                )
    avg_stats = (
        detail_table
        .groupby("ticker")
        .agg(
            avg_rank=("rank", "mean"),
            avg_median_mi=("industry_median_mi", "mean"),
            avg_mean_mi=("industry_mean_mi", "mean"), 
            std_median_mi=("industry_median_mi", "std")
        )
        .reset_index()
    )
    frequency_table = frequency_table.merge(avg_stats, on="ticker", how="left")
    frequency_table = frequency_table.merge(latest, on="ticker", how="left")
    frequency_table = frequency_table.sort_values(["conditional_frequency", "avg_rank", "avg_median_mi"], ascending=[False, True, False])
    frequency_table["mi_cv"] = frequency_table["std_median_mi"] / frequency_table["avg_median_mi"]
    frequency_table["z_score"] = (frequency_table["latest_median_mi"] - frequency_table["avg_median_mi"]) / frequency_table["std_median_mi"]

    cv_cutoff = frequency_table["mi_cv"].median()
    frequency_table["stable_high_mi"] = (frequency_table["conditional_frequency"] >= 0.7) & (frequency_table["avg_rank"] <= 5) & (frequency_table["mi_cv"] <= cv_cutoff)
    frequency_table["current_high_mi"] = frequency_table["avg_rank"] <= 5
    frequency_table["mi_trend_signal"] = "normal"
    frequency_table.loc[frequency_table["z_score"] >= 1, "mi_trend_signal"] = "rising"
    frequency_table.loc[frequency_table["z_score"] <= -1, "mi_trend_signal"] = "falling"

    return detail_table, frequency_table

def compare_high_mi_price_divergence(price_df, high_mi_tickers, frequency_score, ranked, lookback=63):
    result = []
    norm_prices = normalized_price_series(price_df, high_mi_tickers, lookback)
    for ticker in high_mi_tickers:
        ticker_price = norm_prices[ticker]
        peer_prices = norm_prices.drop(columns=[ticker], errors="ignore")
        frequency_component = sum(value for key, value in frequency_score.items() if key in peer_prices.columns)
        norm_freq_component = {key: v / frequency_component for key, v in frequency_score.items() if key in peer_prices.columns}
        mi_component = ranked.set_index("ticker")["industry_median_mi"].reindex(peer_prices.columns).drop(index=ticker, errors="ignore")
        mi_component = mi_component / mi_component.sum()
        raw_weight = (0.5 * pd.Series(norm_freq_component) + 0.5 * mi_component).dropna()
        weight = raw_weight / raw_weight.sum()

        group_avg = peer_prices.mul(weight, axis=1).sum(axis=1)
        weighted_varience = (peer_prices.sub(group_avg, axis=0).pow(2).mul(weight, axis=1)).sum(axis=1)
        group_std = np.sqrt(weighted_varience)

        latest = ticker_price.iloc[-1]
        latest_group_avg = group_avg.iloc[-1]

        if group_std.iloc[-1] == 0 or pd.isna(group_std.iloc[-1]):
                zscore = np.nan
        else:
            zscore = (latest - latest_group_avg) / group_std.iloc[-1]

        result.append({
            "ticker": ticker,
            "normalized_price": latest,
            "group_avg": latest_group_avg,
            "group_std": group_std.iloc[-1],
            "price_z_score": zscore,
            "relative_gap": latest - latest_group_avg,
            "relative_gap_pct": (latest / latest_group_avg) - 1 if latest_group_avg != 0 else np.nan, 
        })

    return pd.DataFrame(result).sort_values("relative_gap_pct").reset_index(drop=True)

def compare_high_mi_return_divergence(returns_df, high_mi_tickers, frequency_score, ranked, lookback=63):
    result = []
    for ticker in high_mi_tickers:
        ticker_returns = returns_df[ticker]
        peer_returns = returns_df[high_mi_tickers].dropna(how="all").drop(columns=[ticker])
        frequency_component = sum(value for key, value in frequency_score.items() if key in peer_returns.columns)
        norm_freq_component = {key: v / frequency_component for key, v in frequency_score.items() if key in peer_returns.columns}
        mi_component = ranked.set_index("ticker")["industry_median_mi"].reindex(peer_returns.columns).drop(index=ticker, errors="ignore")
        mi_component = mi_component / mi_component.sum()
        raw_weight = (0.5 * pd.Series(norm_freq_component) + 0.5 * mi_component).dropna()
        weight = raw_weight / raw_weight.sum()

        group_avg = peer_returns.mul(weight, axis=1).sum(axis=1)
        residual = ticker_returns.sub(group_avg, axis=0).dropna(how="all")
        history = residual.iloc[-lookback-1:-1]
        residual_std = history.std(axis=0)
        residual_mean = history.mean(axis=0)

        latest_return = ticker_returns.iloc[-1]
        latest_group_return = group_avg.iloc[-1]
        latest_residual = residual.iloc[-1]

        zscore = (latest_residual - residual_mean) / residual_std if residual_std != 0 else np.nan

        result.append({
            "ticker": ticker,
            "return": latest_return,
            "group_return": latest_group_return,
            "latest_residual": latest_residual,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "return_z_score": zscore,
        })

    return pd.DataFrame(result).sort_values("return_z_score").reset_index(drop=True) 
      



# Verification
def print_industry_mst_connections(distance_df, metadata, sector):
    groups = group_tickers_by_sector_industry(distance_df.columns, metadata)

    if sector not in groups:
        raise ValueError(f"Sector not found: {sector}")

    industries = groups[sector]

    industry_graph = build_industry_connection_network(
        distance_df,
        metadata,
        sector,
    )

    industry_mst = nx.minimum_spanning_tree(
        industry_graph,
        weight="weight",
    )

    print(f"\nIndustry MST connections for sector: {sector}")
    print("-" * 60)

    for industry_a, industry_b, data in industry_mst.edges(data=True):
        pair, dist = closest_stock_pair_between_groups(
            distance_df,
            industries[industry_a],
            industries[industry_b],
        )

        print(
            f"{industry_a} ↔ {industry_b} | "
            f"industry distance = {data['weight']:.4f} | "
            f"closest pair = {pair} | "
            f"pair distance = {dist:.4f}"
        )

def print_nearest_neighbors(distance_df, ticker, top_n=10):
    if ticker not in distance_df.columns:
        print(f"{ticker} not found in distance matrix.")
        return

    nearest = (
        distance_df[ticker]
        .drop(index=ticker, errors="ignore")
        .dropna()
        .sort_values()
        .head(top_n)
    )

    print(f"\nNearest neighbors for {ticker}")
    print("-" * 40)

    for other, dist in nearest.items():
        print(f"{ticker} ↔ {other}: {dist:.4f}")

def compute_overlap_count_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    tickers = returns.columns
    n = len(tickers)

    overlap_matrix = pd.DataFrame(
        np.zeros((n, n)),
        index=tickers,
        columns=tickers
    )

    for i in range(n):
        for j in range(i, n):
            df = pd.concat([returns.iloc[:, i], returns.iloc[:, j]],axis=1).dropna()

            overlap = len(df)

            overlap_matrix.iloc[i, j] = overlap
            overlap_matrix.iloc[j, i] = overlap

    return overlap_matrix

def main():
    symbols, df_stocks = load_or_cache_df_stocks(force_reload=False, min_trading_days=ROLLING_DAYS)
    metadata = load_stock_metadata()
    # for t in symbols:
    #     if metadata[t].get("industry") == "software":
    #         print(t)

    delete_cache_if_older(RETURNS_CACHE, DF_STOCKS_CACHE)
    df_stocks_returns = load_or_compute(
        RETURNS_CACHE,
        lambda: pd.DataFrame({
            symbol: price_to_return(df)
            for symbol, df in zip(symbols, df_stocks)
        })
    )

    # delete_cache_if_older(EMPIRICAL_MI_CACHE, RETURNS_CACHE)
    # empirical_mi_df = load_or_compute(
    #     EMPIRICAL_MI_CACHE,
    #     lambda: compute_MI_matrix_df(df_stocks_returns, min_overlap=ROLLING_DAYS)
    # )
    # # plot_matrix_correlation_heatmap(empirical_mi_df)

    # delete_cache_if_older(PEARSON_CORR_CACHE, RETURNS_CACHE)
    # pearson_corr = load_or_compute(
    #     PEARSON_CORR_CACHE,
    #     lambda: pearson_correlation(df_stocks_returns)
    # )
    # print(f"Correlation between {symbols[0]} and {symbols[1]}:\n {pearson_corr}\n")

    # delete_cache_if_older(GEOM_DIST_CACHE, PEARSON_CORR_CACHE)
    # pearson_dist = load_or_compute(
    #     GEOM_DIST_CACHE,
    #     lambda: pearson_distance(pearson_corr)
    # )
    # # print(f"True geometric distance between {symbols[0]} and {symbols[1]}:\n {pearson_dist}\n")
    
    # delete_cache_if_older(MI_DIST_CACHE, EMPIRICAL_MI_CACHE)
    # mi_dist = load_or_compute(
    #     MI_DIST_CACHE,
    #     lambda: mi_distance(empirical_mi_df)
    # )

    
    # plot_full_sector_network(mi_dist, metadata, "technology")

    
    # mean_mi_table = build_mean_mi_table(empirical_mi_df,metadata)
    # mean_mi_table["driver_type"] = mean_mi_table.apply(classify_mi_driver, axis=1)

    # top_tech = top_mean_mi_by_group(
    #     mean_mi_table,
    #     sector="technology",
    #     top_n=10
    # )
    # print(top_tech[[
    #         "ticker",
    #         "sector",
    #         "industry",
    #         "overall_mean_mi",
    #         "industry_mean_mi",
    #         "sector_ex_industry_mean_mi",
    #         "driver_type"
    #     ]])


    # top_semis = top_mean_mi_by_group(
    #     mean_mi_table,
    #     sector="technology",
    #     industry="semiconductors",
    #     top_n=3
    # )
    # print(top_semis[[
    #         "ticker",
    #         "sector",
    #         "industry",
    #         "overall_mean_mi",
    #         "industry_mean_mi",
    #         "sector_ex_industry_mean_mi",
    #         "driver_type"
    #     ]])
    
    # overlap_matrix = compute_overlap_count_matrix(df_stocks_returns)
    # mean_mi_table = build_mean_mi_table_robust(empirical_mi_df, overlap_matrix, metadata, min_overlap=ROLLING_DAYS)
    delete_cache_if_older(ROLLING_FREQ_CACHE, RETURNS_CACHE)
    delete_cache_if_older(ROLLING_FREQ_CACHE, METADATA_FILE)

    detail_table, frequency_table = load_or_compute(
        ROLLING_FREQ_CACHE,
        lambda: rolling_top_mi_frequency(
                    df_stocks_returns,
                    metadata,
                    industry="semiconductors",
                    window_length=ROLLING_DAYS,
                    step=21,
                    top_n=10,
                    min_overlap=252,
                    back_years=2,
                )
    )
    print(detail_table[detail_table["buy_signal"]])
    # price_df = pd.DataFrame()
    # for i in range(len(df_stocks)):
    #     df = pd.DataFrame({symbols[i]: df_stocks[i]["Adj Close"]})
    #     price_df = pd.concat([price_df, df], axis=1)

    # high_mi_price_divergence = compare_high_mi_price_divergence(price_df, frequency_table.ticker, lookback=63)
    # print(high_mi_price_divergence)
    
    # print(df_stocks_returns.mean(axis=1))
    # high_mi_return_divergence = compare_high_mi_return_divergence(df_stocks_returns, frequency_table.ticker, lookback=63)
    # print(high_mi_return_divergence)

    # Rank by Median MI
    # print("Rank by Median MI")
    # print(
    #     mean_mi_table[mean_mi_table["industry"] == "semiconductors"][
    #         [
    #             "ticker",
    #             "industry_mean_mi",
    #             "industry_median_mi",
    #             # "industry_trimmed_mean_mi",
    #             # "industry_weighted_mean_mi",
    #             "industry_valid_pairs",
    #             "industry_avg_overlap",
    #         ]
    #     ].sort_values("industry_median_mi", ascending=False).head(20)
    # )
    # # Rank by Mean MI
    # print("Rank by Mean MI")
    # print(
    #     mean_mi_table[mean_mi_table["industry"] == "semiconductors"][
    #         [
    #             "ticker",
    #             "industry_mean_mi",
    #             "industry_median_mi",
    #             # "industry_trimmed_mean_mi",
    #             # "industry_weighted_mean_mi",
    #             "industry_valid_pairs",
    #             "industry_avg_overlap",
    #         ]
    #     ].sort_values("industry_mean_mi", ascending=False).head(20)
    # )
    

if __name__ == "__main__":
    main()