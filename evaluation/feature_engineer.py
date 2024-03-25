import polars as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow as pa


class LASSO:
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = Lasso(alpha=alpha)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def coef(self):
        return self.model.coef_

    def intercept(self):
        return self.model.intercept_


class LINEAR:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def coef(self):
        return self.model.coef_

    def intercept(self):
        return self.model.intercept_


import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class Tool:
    @staticmethod
    def show_feature_distribution(df, features, output_path=None, clip=False):
        n_row = len(features) // 10 + 1
        n_col = 10
        if (n_row == 1):
            n_col = 1
            n_row = len(features)
        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=(3 * n_col, 3 * n_row))
        for i, col in tqdm(enumerate(features)):
            if (n_col == 1 or n_row == 1):
                ax = axes[i]
            else:
                ax = axes[i // n_col, i % n_col]  # 获取当前子图的坐标

            data = df[col].sample(n=10000).to_pandas()
            if (clip):
                q999 = data.quantile(0.999)
                q001 = data.quantile(0.001)
                data = data.clip(q001, q999)

            ax.hist(data, bins=100, density=True)  # 绘制分布图
            ax.set_title(col)  # 设置子图标题
        plt.tight_layout()  # 调整子图布局
        if (output_path is not None):
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def r2(y_pred, y_true):
        return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

    @staticmethod
    def mse(y_pred, y_true):
        return ((y_true - y_pred) ** 2).mean()

    @staticmethod
    def rmse(y_pred, y_true):
        return np.sqrt(((y_true - y_pred) ** 2).mean())

    @staticmethod
    def evalation(y_pred, y_true, tag=""):
        matrix = {}
        matrix['R2'] = Tool.r2(y_pred, y_true)
        matrix['RMSE'] = Tool.rmse(y_pred, y_true)
        print(
            f"{tag}\t  R2:{matrix['R2']:.6f}, RMSE:{matrix['RMSE']:.6f}")
        return matrix

    @staticmethod
    def show_correlation(df, features_A, features_B, output_path=None, ax=None):
        if (ax == None):
            fig, ax = plt.subplots(
                figsize=(len(features_B) // 5 + 10, len(features_A) // 5 + 10))
        else:
            fig = ax.get_figure()
        selected_features = list(set(features_A + features_B))
        if (len(selected_features) > 1):
            df_selected = df.select(selected_features)
            corr_matrix = df_selected.corr()
            corr_matrix = corr_matrix.to_pandas()
            corr_matrix.set_index(corr_matrix.columns, inplace=True)
            corr = corr_matrix.loc[features_A, features_B]
            corr_log = corr.abs()
        else:
            corr_log = pd.DataFrame(1, index=features_A, columns=features_B)
        cax = ax.matshow(corr_log, cmap='coolwarm', vmin=0, vmax=1)
        cbar = fig.colorbar(cax)
        ax.set_xticks(range(len(features_B)))
        ax.set_xticklabels(features_B, rotation=90)
        ax.set_yticks(range(len(features_A)))
        ax.set_yticklabels(features_A)
        if (output_path is not None):
            if (output_path.endswith('.png')):
                plt.savefig(output_path)
                plt.close()
        else:
            plt.show()


features_new = []


def feature_engineering(data, num=5):
    data = data.fill_null(0)
    data = data.fill_nan(0)
    features = []
    targets = []

    # for i in range(15):
    #     data = data.with_columns([
    #         (pl.col(f'askRate{i}') - (pl.col('bidRate0')+pl.col('askRate0'))/2).alias(f'nom_Rate{i}') ])
    #     features.append(f'nom_Rate{i}')

    # for i in range(15):
    #     data = data.with_columns([
    #         (pl.col(f'bidSize{i}').log1p()).alias(f'log_bid{i}'),
    #         (pl.col(f'askSize{i}').log1p()).alias(f'log_ask{i}')
    #     ])
    #     features.append(f'log_bid{i}')
    #     features.append(f'log_ask{i}')
    count = num
    for i in range(1):
        data = data.with_columns(
            [
                ((pl.col(f'askRate{i}') + pl.col(f'bidRate{i}')) / 2).alias(f'tmp.mid_price{i}'),
                ((pl.col(f'askRate{i}') * pl.col(f'bidSize{i}') + pl.col(f'bidRate{i}') * pl.col(f'askSize{i}')) / (
                        pl.col(f'bidSize{i}') + pl.col(f'askSize{i}'))).alias(f'tmp.smart_mid_price{i}')
            ]
        )

    for i in range(15):
        if (i == 0):
            data = data.with_columns(
                [
                    pl.col(f'bidSize{i}').alias(f'tmp.pre_sum_bid_size{i}'),
                    pl.col(f'askSize{i}').alias(f'tmp.pre_sum_ask_size{i}')
                ]
            )
        else:
            data = data.with_columns(
                [
                    (pl.col(f'tmp.pre_sum_bid_size{i - 1}') + pl.col(
                        f'bidSize{i}')).alias(f'tmp.pre_sum_bid_size{i}'),
                    ((pl.col(f'tmp.pre_sum_ask_size{i - 1}') + pl.col(
                        f'askSize{i}'))).alias(f'tmp.pre_sum_ask_size{i}')
                ])

    # BP SIZE
    for i in range(count):
        data = data.with_columns(
            ((pl.col(f'tmp.pre_sum_bid_size{i}') - pl.col(f'tmp.pre_sum_ask_size{i}'))
             / (pl.col(f'tmp.pre_sum_bid_size{i}') + pl.col(f'tmp.pre_sum_ask_size{i}')))
            .alias(f'fea.BP{i}')
        )
        features.append(f'fea.BP{i}')

    for i in range(count):
        data = data.with_columns(
            [
                ((pl.col(f'askSize{i}') / pl.col(f'askSize{i}').mean()).log()).alias(f'fea.log_ask_size{i}'),
                ((pl.col(f'bidSize{i}') / pl.col(f'askSize{i}').mean()).log()).alias(f'fea.log_bid_size{i}')
            ]
        )
        features.append(f'fea.log_ask_size{i}')
        features.append(f'fea.log_bid_size{i}')

    # OI
    for level in range(count):
        cond1 = (pl.col(f'askRate{level}') > pl.col(f'askRate{level}').shift(1)).alias('cond1')
        cond2 = (pl.col(f'askRate{level}') == pl.col(f'askRate{level}').shift(1)).alias('cond2')
        # Calculate the new column values based on the conditions
        delVA = pl.when(cond1 | pl.col(f'askRate{level}').shift().is_null()).then(0).otherwise(
            pl.when(cond2).then(pl.col(f'askSize{level}').diff()).otherwise(pl.col(f'askSize{level}'))).alias('delVA')

        # Define the conditions for the new column
        cond1 = (pl.col(f'bidRate{level}') < pl.col(f'bidRate{level}').shift(1)).alias('cond1')
        cond2 = (pl.col(f'bidRate{level}') == pl.col(f'bidRate{level}').shift(1)).alias('cond2')
        # Calculate the new column values based on the conditions
        delVB = pl.when(cond1 | pl.col(f'bidRate{level}').shift().is_null()).then(0).otherwise(
            pl.when(cond2).then(pl.col(f'bidSize{level}').diff()).otherwise(pl.col(f'bidSize{level}'))).alias('delVB')

        mkd1 = np.sqrt(pl.sum_horizontal([pl.col(f'askSize{i}') for i in range(level + 1)]) + pl.sum_horizontal(
            [pl.col(f'bidSize{i}') for i in range(level + 1)]))
        data = data.with_columns(((delVB - delVA) / mkd1).alias(f'fea.OI{level}'))

        features.append(f'fea.OI{level}')

    # NOM_RATE
    for i in range(count):
        data = data.with_columns(
            [
                (pl.col(f'askRate{i}') / pl.col(f'tmp.mid_price0') - 1).alias(f'fea.nom_ask_prc_{i}'),
                (pl.col(f'bidRate{i}') / pl.col(f'tmp.mid_price0') - 1).alias(f'fea.nom_bid_prc_{i}'),
            ]
        )
        features.append(f'fea.nom_ask_prc_{i}')
        features.append(f'fea.nom_bid_prc_{i}')

    # PP
    for i in range(count):
        data = data.with_columns(
            [
                ((pl.col(f'askRate{i}') + pl.col(f'bidRate{i}') - 2 * pl.col(f'tmp.smart_mid_price0')) / (
                            pl.col(f'askRate{i}') - pl.col(f'bidRate{i}'))).alias(f'fea.PP{i}')
            ]
        )
        features.append(f'fea.PP{i}')

    # B_RET
    # for i in [1,2,4,8,16,32]:
    #     data = data.with_columns(
    #             (pl.col(f'tmp.mid_price0').shift(i) / pl.col(f'tmp.mid_price0') - 1).alias(f'fea.back_ret{i}')
    #     )
    #     data = data.with_columns(
    #             pl.when(pl.col(f'tmp.mid_price0')<0).then(-pl.col(f'fea.back_ret{i}')).otherwise(pl.col(f'fea.back_ret{i}')).alias(f'fea.back_ret{i}')
    #     )

    #     features.append(f'fea.back_ret{i}')

    # F_RET
    for i in [5, 10, 50, 90, 100, 110, 150, 300]:
        data = data.with_columns(
            (pl.col(f'tmp.mid_price0').shift(-i) / pl.col(f'tmp.mid_price0') - 1).alias(f'target.forw_ret{i}') * 1000
        )
        data = data.with_columns(
            pl.when(pl.col(f'tmp.mid_price0') < 0).then(-pl.col(f'target.forw_ret{i}')).otherwise(
                pl.col(f'target.forw_ret{i}')).alias(f'target.forw_ret{i}')
        )

        targets.append(f'target.forw_ret{i}')

    data = data.fill_null(0)
    data = data.fill_nan(0)
    return data, features, targets


def normal_feature(df, feature):
    q001 = df[feature].quantile(0.01)
    q999 = df[feature].quantile(0.99)

    df = df.with_columns(
        pl.when(pl.col(feature) < q001).then(
            q001).otherwise(pl.col(feature)).alias(feature)
    )
    df = df.with_columns(
        pl.when(pl.col(feature) > q999).then(
            q999).otherwise(pl.col(feature)).alias(feature)
    )

    std = df[feature].std()
    # mean = df[feature].mean()
    mean = 0
    df = df.with_columns(
        ((pl.col(feature) - mean) / std).alias(feature)
    )

    if ("log" in feature or "nom" in feature):
        min = pl.col(feature).min()
        max = pl.col(feature).max()
        df = df.with_columns(
            ((pl.col(feature) - min) / (max - min) * 2).alias(feature)
        )
    return df


def normal_target(df, feature):
    df = df.with_columns(
        pl.when(pl.col(feature) < -5.0).then(
            -5.0).otherwise(pl.col(feature)).alias(feature)
    )
    df = df.with_columns(
        pl.when(pl.col(feature) > 5).then(
            5.0).otherwise(pl.col(feature)).alias(feature)
    )

    return df


def shift_feature(df, feature, shift):
    df = df.with_columns(
        pl.col(feature).shift(shift).alias(f'{feature}_shift{shift}')
    )
    return df, f'{feature}_shift{shift}'


def get_features(input_data):
    data, features, targets = feature_engineering(input_data.clone())
    for feature in tqdm(features):
        data = normal_feature(data, feature)

    shift_features = []
    for feature in tqdm(features):
        if ("nom" not in feature):
            for shift in [1, 2, 4, 8, 16, 32]:
                data, new_feature = shift_feature(data, feature, shift)
                shift_features.append(new_feature)
    features += shift_features

    data = data.fill_null(0)
    data = data.fill_nan(0)
    return data, features, targets


def normal_test_feature(df, feature, train_std, train_min, train_max, q001, q999):

    df = df.with_columns(
        pl.when(pl.col(feature) < q001).then(
            q001).otherwise(pl.col(feature)).alias(feature)
    )
    df = df.with_columns(
        pl.when(pl.col(feature) > q999).then(
            q999).otherwise(pl.col(feature)).alias(feature)
    )
    # mean = df[feature].mean()
    mean = 0
    df = df.with_columns(
        ((pl.col(feature) - mean) / train_std).alias(feature)
    )

    if ("log" in feature or "nom" in feature):
        df = df.with_columns(
            ((pl.col(feature) - train_min) / (train_max - train_min) * 2).alias(feature)
        )
    return df

def normal_feature(df, feature):
    q001 = df[feature].quantile(0.01)
    q999 = df[feature].quantile(0.99)

    df = df.with_columns(
        pl.when(pl.col(feature) < q001).then(
            q001).otherwise(pl.col(feature)).alias(feature)
    )
    df = df.with_columns(
        pl.when(pl.col(feature) > q999).then(
            q999).otherwise(pl.col(feature)).alias(feature)
    )

    std = df[feature].std()
    # mean = df[feature].mean()
    mean = 0
    df = df.with_columns(
        ((pl.col(feature) - mean) / std).alias(feature)
    )

    if ("log" in feature or "nom" in feature):
        min = pl.col(feature).min()
        max = pl.col(feature).max()
        df = df.with_columns(
            ((pl.col(feature) - min) / (max - min) * 2).alias(feature)
        )
        return df, q001, q999, std, min, max
    return df, q001, q999, std, 0, 2


def normal_target(df, feature):
    df = df.with_columns(
        pl.when(pl.col(feature) < -5.0).then(
            -5.0).otherwise(pl.col(feature)).alias(feature)
    )
    df = df.with_columns(
        pl.when(pl.col(feature) > 5).then(
            5.0).otherwise(pl.col(feature)).alias(feature)
    )

    return df


def shift_feature(df, feature, shift):
    df = df.with_columns(
        pl.col(feature).shift(shift).alias(f'{feature}_shift{shift}')
    )
    return df, f'{feature}_shift{shift}'


def get_features(input_data, num=5, is_shift=True):
    data, features, targets = feature_engineering(input_data.clone(), num)

    # for feature in tqdm(features):
    #    data,_,_,_ = normal_feature(data, feature)
    if is_shift:
        shift_features = []
        for feature in tqdm(features):
            if ("nom" not in feature):
                for shift in [1, 2, 4, 8, 16, 32]:
                    data, new_feature = shift_feature(data, feature, shift)
                    shift_features.append(new_feature)
        features += shift_features

    data = data.fill_null(0)
    data = data.fill_nan(0)
    return data, features, targets


def rev_data_augmentation(org_data):
    data = org_data.clone()
    rename_dict = {}
    for i in range(15):
        rename_dict[f'bidRate{i}'] = f'askRate{i}'
        rename_dict[f'askRate{i}'] = f'bidRate{i}'
        rename_dict[f'bidSize{i}'] = f'askSize{i}'
        rename_dict[f'askSize{i}'] = f'bidSize{i}'

    data = data.rename(rename_dict)

    for i in range(15):
        data = data.with_columns(
            [(-pl.col(f'askRate{i}')).alias(f'askRate{i}'),
             (-pl.col(f'bidRate{i}')).alias(f'bidRate{i}')]
        )

    if ('y' in data.columns):
        data = data.with_columns(
            (-pl.col('y')).alias('y')
        )

    return data


def shift_rate_augmentation(org_data, shift):
    data = org_data.clone()
    for i in range(15):
        data = data.with_columns(
            [
                (pl.col(f'bidRate{i}') + shift).alias(f'bidRate{i}'),
                (pl.col(f'askRate{i}') + shift).alias(f'askRate{i}'),
            ]
        )
    return data


# def shift_size_augmentation(org_data, shift):
#     data = org_data.clone()
#     for i in range(15):
#         data = data.with_columns(
#             [
#                 (pl.col(f'bidSize{i}') * shift).alias(f'bidSize{i}'),
#                 (pl.col(f'askSize{i}') * shift).alias(f'askSize{i}'),
#             ]
#         )
#     return data

def data_set_split(org_data, names, test_size):
    train_data = {}
    test_data = {}
    for name in names:
        train_data[name], test_data[name] = train_test_split(
            org_data[name], test_size=test_size, shuffle=False)
    return train_data, test_data


def data_set_concat(org_data, names, labels):
    data = pl.concat([org_data[name][labels] for name in names])
    return data


import lightgbm as lgb
import argparse
import polars as pl


def get_augment_data(org_data):
    aug_features = []
    aug_features_flat = []
    rev_org_data = rev_data_augmentation(org_data)
    rev_data, rev_features, rev_art_targets = get_features(rev_org_data)
    aug_features.append(rev_features)
    for feature in rev_features:
        aug_features_flat.append(feature)

    shift_10_org_data = shift_rate_augmentation(org_data, 10)
    shift_10_data, shift_10_features, shift_10_art_targets = get_features(shift_10_org_data)
    aug_features.append(shift_10_features)

    shift_rev_10_org_data = shift_rate_augmentation(rev_org_data, -10)
    shift_rev_10_data, shift_rev_10_features, shift_rev_10_art_targets = get_features(shift_rev_10_org_data)
    aug_features.append(shift_rev_10_features)

    augment_data = {}
    augment_data['rev_data'] = rev_data
    augment_data['shift_10_data'] = shift_10_data
    augment_data['shift_rev_10_data'] = shift_rev_10_data
    return augment_data, aug_features, aug_features_flat