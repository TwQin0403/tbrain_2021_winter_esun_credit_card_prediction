import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

import functools

import re

import sys

sys.path.insert(0, './code')

from utils import DataLogger  # noqa: E402


class DataNotFoundException(Exception):
    pass


def get_time_split(df):
    df_12 = df[df['dt'] <= 12]
    df_16 = df[(df['dt'] > 12) & (df['dt'] <= 16)]
    # df_20 = df[(df['dt'] > 16) & (df['dt'] <= 19)]
    # df_21 = df[(df['dt'] > 17) & (df['dt'] <= 20)]
    df_22 = df[(df['dt'] > 18) & (df['dt'] <= 21)]
    df_23 = df[(df['dt'] > 19) & (df['dt'] <= 22)]
    # df_24 = df[(df['dt'] > 20) & (df['dt'] <= 23)]
    # df_25 = df[(df['dt'] > 21) & (df['dt'] <= 24)]
    r_dict = {
        "one_to_twelve": df_12,
        "twelve_to_sixteen": df_16,
        # "prev_three_months_20": df_20,
        # "prev_three_months_21": df_21,
        "prev_three_months_22": df_22,
        "prev_three_months_23": df_23,
        # "prev_three_months_24": df_24,
        # "prev_three_months_25": df_25
    }
    return r_dict


def get_merge_dict():
    merge_dict = {
        # 20: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_20"],
        # 21: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_21"],
        22: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_22"],
        23: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_23"],
        # 24: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_24"],
        # 25: ["one_to_twelve", "twelve_to_sixteen", "prev_three_months_25"],
    }
    return merge_dict


def get_time_split_result(a_func):
    @functools.wraps(a_func)
    def wrapper(self, df):
        r_dict = defaultdict(list)
        df_dict = get_time_split(df)
        use_dict = {key: a_func(self, df_dict[key]) for key in df_dict.keys()}
        merge_dict = get_merge_dict()
        for dt in merge_dict.keys():
            vals_12 = use_dict[merge_dict[dt][0]]
            vals_16 = use_dict[merge_dict[dt][1]]
            vals_prevs = use_dict[merge_dict[dt][2]]
            for val, val_12, val_16 in zip(vals_prevs, vals_12, vals_16):
                name = val[0]
                name_12 = "{}_12".format(name)
                name_16 = "{}_16".format(name)
                r_dict[name].append(val[1])
                r_dict[name_12].append(val_12[1])
                r_dict[name_16].append(val_16[1])
        return r_dict

    return wrapper


class DataLoader():
    def __init__(self):
        self.output_path = Path(os.path.abspath(os.getcwd())) / 'output'
        self.input_path = Path(os.path.abspath(os.getcwd())) / 'input'
        self.model_path = Path(os.path.abspath(os.getcwd())) / 'model'

    def save_data(self, cls, data_name, message):
        logger = DataLogger()
        logger.save_data("Save data {} is generated from {}".format(
            data_name, message))
        joblib.dump(cls, self.output_path / data_name)
        logger.save_data("{} is sucessfully saved".format(data_name))

    def load_data(self, data_name, data_type='joblib', **kwargs):
        if data_type == 'joblib':
            data = joblib.load(self.input_path / data_name, **kwargs)
        elif data_type == 'csv':
            data = pd.read_csv(self.input_path / data_name, **kwargs)
        return data

    def load_result(self, data_name, data_type='joblib', **kwargs):
        if data_type == 'joblib':
            data = joblib.load(self.output_path / data_name, **kwargs)
        elif data_type == 'csv':
            data = pd.read_csv(self.output_path / data_name, **kwargs)
        return data


class FeatLoader(DataLoader):
    def __init__(self):
        super(FeatLoader, self).__init__()
        self.required_cate = ('2', '6', '10', '12', '13', '15', '18', '19',
                              '21', '22', '25', '26', '36', '37', '39', '48')
        self.shop_cate = [str(i + 1) for i in range(48)] + ['other']
        self.shop_amt = [
            "shop_{}_amt".format(shop_tag) for shop_tag in self.shop_cate
        ]
        self.shop_cnt = [
            "shop_{}_cnt".format(shop_tag) for shop_tag in self.shop_cate
        ]
        self.card_cate = [str(i + 1) for i in range(14)] + ['other']
        self.card_amt = [
            "card_{}_txn_amt".format(card_cate) for card_cate in self.card_cate
        ]
        self.card_cnt = [
            "card_{}_txn_cnt".format(card_cate) for card_cate in self.card_cate
        ]
        self.count = 0
        self.profile_cate = [
            "masts",
            "educd",
            "trdtp",
            "naty",
            "poscd",
            "cuorg",
            "primary_card",
            "slam",
            "age",
            "gender_code",
        ]
        self.basic_info = [
            'masts',
            'educd',
            'trdtp',
            'naty',
            'poscd',
            'cuorg',
            'primary_card',
            'age',
            'gender_code',
        ]
        self.dts = [dt for dt in range(1, 25)]

    def update_data(self, data):
        self.data = data.copy()


class AmtFeatLoader(FeatLoader):
    def __init__(self):
        super(AmtFeatLoader, self).__init__()
        self.get_feat_config()

    def update_a_df(self, df):
        result = {'chid': df['chid'].iloc[0]}
        if self.count % 10000 == 0:
            print(result)
        self.count += 1

        for feat_func in self.feat_config:
            result.update(feat_func(df))

        # result = pd.DataFrame(result)
        return result

    def get_feat_config(self):
        self.feat_config = {self.get_amt_by_months}

    def get_amt_by_months(self, df):
        def get_shop_amt_cate(x):
            dt, shop_tag = x
            name = "shop_{}_amt_{}".format(shop_tag, dt)
            return name

        result = {}
        for dt in range(1, 25):
            for shop_amt_cate in self.shop_amt:
                result.update({shop_amt_cate + '_{}'.format(dt): 0})
        if df.empty:
            return result

        else:
            df['shop_amt_cate'] = df[['dt',
                                      'shop_tag']].apply(get_shop_amt_cate,
                                                         axis=1)
            amt_dict = {
                shop_amt_cate: amt
                for amt, shop_amt_cate in zip(df['txn_amt'],
                                              df['shop_amt_cate'])
            }
            result.update(amt_dict)
        return result

    def fit(self):
        if not hasattr(self, 'data'):
            raise DataNotFoundException("Data not found! Please update data")
        df_group = self.data.groupby(['chid'])
        df_group = [df[1] for df in df_group]
        pool = Pool(8, maxtasksperchild=1000)
        feat_group = pool.map(self.update_a_df, df_group)
        pool.close()
        self.feats = pd.DataFrame(feat_group)


class ProfileFeatLoader(FeatLoader):
    def __init__(self):
        super(ProfileFeatLoader, self).__init__()
        self.get_feat_config()
        self.card_cnt_pct = [
            "card_{}_cnt_pct".format(cate) for cate in self.card_cate
        ]
        self.card_avg_amt = [
            "card_{}_avg_amt".format(cate) for cate in self.card_cate
        ]

    def fit(self):
        # run 500000 times loop
        if not hasattr(self, 'data'):
            raise DataNotFoundException("Data not found! Please update data")
        self.data = self.get_early_calculation(self.data)
        df_group = self.data.groupby(['chid'])
        df_group = [df[1] for df in df_group]
        pool = Pool(8, maxtasksperchild=1000)
        feat_group = pool.map(self.update_a_df, df_group)
        pool.close()
        self.feats = pd.concat(feat_group)

    def get_early_calculation(self, df):
        df['avg_amt'] = df['txn_amt'] / df['txn_cnt']
        df['offline_cnt_pct'] = df['txn_cnt'] / (df['domestic_offline_cnt'] +
                                                 df['overseas_offline_cnt'])
        df['online_cnt_pct'] = df['txn_cnt'] / (df['domestic_online_cnt'] +
                                                df['overseas_online_cnt'])
        df['domestic_cnt_pct'] = df['txn_cnt'] / (df['domestic_offline_cnt'] +
                                                  df['domestic_online_cnt'])
        df['overseas_cnt_pct'] = df['txn_cnt'] / (df['overseas_offline_cnt'] +
                                                  df['overseas_online_cnt'])
        # generate card amt
        for cate in self.card_cate:
            df['card_{}_txn_amt'.format(
                cate)] = df['card_{}_txn_amt_pct'.format(cate)] * df['txn_amt']
        # generate card cnt ratio
        for cate in self.card_cate:
            new_key = "card_{}_cnt_pct".format(cate)
            cnt_key = "card_{}_txn_cnt".format(cate)
            df[new_key] = df[cnt_key] / df['txn_cnt']

        # generate the avg for card cate
        for cate in self.card_cate:
            new_key = "card_{}_avg_amt".format(cate)
            amt_key = "card_{}_txn_amt".format(cate)
            cnt_key = "card_{}_txn_cnt".format(cate)
            df[new_key] = df[amt_key] / df[cnt_key]

        return df

    def update_a_df(self, df):
        # df: user history records
        result = {
            'dt': [22, 23],
            'chid': [df['chid'].iloc[0]] * 2,
        }
        if self.count % 10000 == 0:
            print(result)
        self.count += 1
        for feat_func in self.feat_config:
            result.update(feat_func(df))

        result = pd.DataFrame(result)
        return result

    def get_feat_config(self):
        self.feat_config = {
            # 最開始使用信用卡時間 #首刷月
            # 離首刷月多久
            self.get_start_use_dt,
            # # 消費多少種類
            # # 消費多少重要種類
            self.get_how_many_tags,
            # # basic info
            self.get_basic_profile,
        }

    def get_basic_profile(self, df):
        if df.empty:
            r_dict = {
                profile_cate: [-1] * 3
                for profile_cate in self.profile_cate
            }
        else:
            r_dict = {
                profile_cate: df[profile_cate].iloc[0]
                for profile_cate in self.profile_cate
            }
        return r_dict

    @get_time_split_result
    def get_how_many_tags(self, df):
        if df.empty:
            r_list = [("how_many_tag", -1), ("how_many_tag_imp", -1)]
        else:
            how_many_tag = len(df['shop_tag'].unique())
            how_many_tag_imp = len(df[df['shop_tag'].isin(
                self.required_cate)]['shop_tag'].unique())
            r_list = [("how_many_tag", how_many_tag),
                      ("how_many_tag_imp", how_many_tag_imp)]
        return r_list

    def get_start_use_dt(self, df):
        if df.empty:
            r_dict = {"start_dt": [-1] * 2, "how_long_dt": [-1] * 2}

        else:
            start_dt = df['dt'].iloc[0]
            how_long_dt = np.array([24, 25]) - np.array([start_dt] * 2)

            r_dict = {
                "start_dt": [start_dt] * 2,
                "how_long_dt": list(how_long_dt)
            }
        return r_dict



class CntFeatLoader(FeatLoader):
    def __init__(self):
        super(CntFeatLoader, self).__init__()
        self.get_feat_config()

    def get_feat_config(self):
        self.feat_config = {self.get_cnt_by_months}

    def get_cnt_by_months(self, df):
        def get_shop_cnt_cate(x):
            dt, shop_tag = x
            name = "shop_{}_cnt_{}".format(shop_tag, dt)
            return name

        result = {}
        for dt in range(1, 25):
            for shop_cnt_cate in self.shop_cnt:
                result.update({shop_cnt_cate + '_{}'.format(dt): 0})
        if df.empty:
            return result

        else:
            df['shop_cnt_cate'] = df[['dt',
                                      'shop_tag']].apply(get_shop_cnt_cate,
                                                         axis=1)
            cnt_dict = {
                shop_cnt_cate: cnt
                for cnt, shop_cnt_cate in zip(df['txn_cnt'],
                                              df['shop_cnt_cate'])
            }
            result.update(cnt_dict)
        return result

    def update_a_df(self, df):
        result = {'chid': df['chid'].iloc[0]}
        if self.count % 10000 == 0:
            print(result)
        self.count += 1

        for feat_func in self.feat_config:
            result.update(feat_func(df))

        return result

    def fit(self):
        if not hasattr(self, 'data'):
            raise DataNotFoundException("Data not found! Please update data")

        df_group = self.data.groupby(['chid'])
        df_group = [df[1] for df in df_group]
        pool = Pool(8, maxtasksperchild=1000)
        feat_group = pool.map(self.update_a_df, df_group)
        pool.close()
        self.feats = pd.DataFrame(feat_group)


class RankTopFeatLoader(FeatLoader):
    def __init__(self):
        super(RankTopFeatLoader, self).__init__()
        self.get_feat_config()
        self.shop_cate_map = {
            i: a_shop_cate
            for i, a_shop_cate in enumerate(self.shop_cate)
        }
        self.imp_cate_map = {
            i: imp_cate
            for i, imp_cate in enumerate(self.required_cate)
        }

    def update_a_df(self, df):
        print(df.columns[0])
        result = []
        for feat_func in self.feat_config:
            result.append(feat_func(df))
        tops = pd.concat(result, axis=1)
        return tops

    def get_feat_config(self):
        self.feat_config = [
            self.get_tops_by_months,
            self.get_imp_tops_by_months,
        ]

    def get_tops_by_months(self, df):
        dt = df.columns[0].split('_')[-1]
        top3 = df.apply(lambda x: np.argsort(x), axis=1).iloc[:, -3:]
        top3.columns = [
            'top3_{}'.format(dt), 'top2_{}'.format(dt), 'top1_{}'.format(dt)
        ]
        for col in top3.columns:
            top3[col] = top3[col].map(self.shop_cate_map)
        top3['how_many_cate_{}'.format(dt)] = df.gt(0).sum(axis=1)
        top3.loc[
            top3['how_many_cate_{}'.format(dt)] == 0,
            ['top3_{}'.format(dt), 'top2_{}'.format(dt), 'top1_{}'.
             format(dt)]] = "-1"
        top3.loc[top3['how_many_cate_{}'.format(dt)] == 1,
                 ['top3_{}'.format(dt), 'top2_{}'.format(dt)]] = "-1"
        top3.loc[top3['how_many_cate_{}'.format(dt)] == 2,
                 ['top3_{}'.format(dt)]] = "-1"
        return top3

    def get_imp_tops_by_months(self, df):
        dt = df.columns[0].split('_')[-1]
        reg = r"shop_(\d+_|other_)(.+)_\d+"
        fetch_type = re.findall(reg, df.columns[0])[0][1]
        imp_cols = [
            "shop_{}_{}_{}".format(a_cate, fetch_type, dt)
            for a_cate in self.required_cate
        ]
        imp_df = df[imp_cols].copy()
        imp_top3 = imp_df.apply(lambda x: np.argsort(x), axis=1).iloc[:, -3:]
        imp_top3.columns = [
            'imp_top3_{}'.format(dt), 'imp_top2_{}'.format(dt),
            'imp_top1_{}'.format(dt)
        ]

        for col in imp_top3.columns:
            imp_top3[col] = imp_top3[col].map(self.imp_cate_map)
        imp_top3['how_many_cate_imp_{}'.format(dt)] = imp_df.gt(0).sum(axis=1)
        imp_top3.loc[imp_top3["how_many_cate_imp_{}".format(dt)] == 0, [
            "imp_top3_{}".format(dt), "imp_top2_{}".format(dt), "imp_top1_{}".
            format(dt)
        ]] = "-1"
        imp_top3.loc[
            imp_top3["how_many_cate_imp_{}".format(dt)] == 1,
            ["imp_top3_{}".format(dt), "imp_top2_{}".format(dt)]] = "-1"
        imp_top3.loc[imp_top3['how_many_cate_imp_{}'.format(dt)] == 2,
                     ['imp_top3_{}'.format(dt)]] = "-1"
        return imp_top3

    def fit(self):
        if not hasattr(self, 'data'):
            raise DataNotFoundException("Data not found! Please update data")
        feats = [self.data[['chid']].reset_index(drop=True)]
        df = self.data.drop("chid", axis=1).reset_index(drop=True)
        cols = list(df.columns)
        cols_group = [cols[dt * 49:(1 + dt) * 49] for dt in range(24)]
        df_group = [df[col_seg] for col_seg in cols_group]
        pool = Pool(4, maxtasksperchild=1000)
        feat_group = pool.map(self.update_a_df, df_group)
        pool.close()
        self.feats = pd.concat(feats + feat_group, axis=1)


class ProfileShopFeatLoader(FeatLoader):
    def __init__(self):
        super(ProfileShopFeatLoader, self).__init__()
        self.get_feat_config()

    def update_data(self, data):
        self.data = {}
        for dt in range(1, 25):
            use_cols = []
            for shop_tag in self.required_cate:
                col = "shop_{}_amt_{}".format(shop_tag, dt)
                use_cols.append(col)
            tmp_df = data[['chid'] + use_cols]
            tmp_df.columns = ['chid'] + list(self.required_cate)
            tmp_df['dt'] = dt
            tmp_df['query_id'] = tmp_df['chid'].apply(
                lambda x: str(x)) + tmp_df['dt'].apply(lambda x: str(x))
            tmp_df = tmp_df.melt(id_vars=['chid', 'dt', 'query_id'],
                                 var_name='shop_tag',
                                 value_name='txn_amt')
            self.data.update({dt: tmp_df})

    def get_feat_config(self):
        self.feat_config = {self.get_rank_by_months}

    def get_rank_by_months(self, dt):
        print(dt)
        df = self.data[dt]
        df_group = df.groupby('query_id')
        df_group = [df[1] for df in df_group]
        df_group = [
            df.sort_values(by='txn_amt', ascending=False) for df in df_group
        ]
        df = pd.concat(df_group)
        df['rank_{}'.format(dt)] = list(range(1, 17)) * 500000
        df['rank_{}'.format(dt)] = df[['rank_{}'.format(dt)
                                       ]].mask(df['txn_amt'] == 0, 0)
        df = df[['chid', 'shop_tag', 'rank_{}'.format(dt)]]
        return df

    def fit(self):
        feat_group = [self.get_rank_by_months(1)]
        for dt in range(2, 25):
            feat_group.append(self.get_rank_by_months(dt))
        self.feats = pd.concat(feat_group, axis=1)


class FreqFeatLoader(FeatLoader):
    def __init__(self, prefix, fetch_type):
        super(FreqFeatLoader, self).__init__()
        self.prefix = prefix
        self.fetch_type = fetch_type

    def update_data(self, data):
        self.get_cols(self.prefix, self.fetch_type)
        self.train_23 = data.drop(self.cols_24 + self.cols_23,
                                  axis=1).reset_index(drop=True)
        self.train_24 = data.drop(self.cols_1 + self.cols_24,
                                  axis=1).reset_index(drop=True)
        self.test = data.drop(self.cols_1 + self.cols_2,
                              axis=1).reset_index(drop=True)
        self.train_23.columns = self.get_new_cols(self.train_23, 23)
        self.train_24.columns = self.get_new_cols(self.train_24, 24)
        self.test.columns = self.get_new_cols(self.test, 25)

        self.train_23['dt'] = 23
        self.train_24['dt'] = 24
        self.test['dt'] = 25

        self.train_23['query_id'] = self.train_23['chid'].apply(
            lambda x: str(x)) + self.train_23['dt'].apply(lambda x: str(x))
        self.train_24['query_id'] = self.train_24['chid'].apply(
            lambda x: str(x)) + self.train_24['dt'].apply(lambda x: str(x))
        self.test['query_id'] = self.test['chid'].apply(
            lambda x: str(x)) + self.test['dt'].apply(lambda x: str(x))

    def get_cols(self, prefix='amt', fetch_type='total'):
        if fetch_type == 'total':
            use_cates = self.shop_cate
        else:
            use_cates = self.required_cate
        self.cols_24 = [
            "shop_{}_{}_24".format(shop_cate, self.prefix)
            for shop_cate in use_cates
        ]
        self.cols_23 = [
            "shop_{}_{}_23".format(shop_cate, self.prefix)
            for shop_cate in use_cates
        ]
        self.cols_1 = [
            "shop_{}_{}_1".format(shop_cate, self.prefix)
            for shop_cate in use_cates
        ]
        self.cols_2 = [
            "shop_{}_{}_2".format(shop_cate, self.prefix)
            for shop_cate in use_cates
        ]
        self.cols_less_3 = []
        for use_cate in use_cates:
            for dt in range(1, 4):
                col = "shop_{}_{}_{}".format(use_cate, self.prefix, dt - 4)
                self.cols_less_3.append(col)

    def get_new_cols(self, df, dt):
        new_cols = ['chid']
        for col in df.columns[1:]:
            reg = r"(.+_)\d+"
            c_dt = col.split('_')[-1]
            new_dt = int(c_dt) - dt
            n_col = re.findall(reg, col)[0]
            n_col = n_col + str(new_dt)
            new_cols.append(n_col)
        return new_cols

    def get_freq_buy(self, df, start_dt=1):
        results = {
            'chid': df['chid'].to_list(),
            'dt': df['dt'].to_list(),
            'query_id': df['query_id'].to_list()
        }
        for shop_cate in self.required_cate:
            cols = [
                "shop_{}_{}_{}".format(shop_cate, self.prefix, dt - 23)
                for dt in range(start_dt, 23)
            ]
            shop_ratio = df[cols][df[cols] != 0].count(1) / 22
            results[shop_cate] = shop_ratio

        feats = pd.DataFrame(results)
        feats = feats.melt(id_vars=['chid', 'dt', 'query_id'],
                           var_name='shop_tag',
                           value_name='freq_buy')
        return feats

    def get_relative_freq_buy(self, df, start_dt=1):
        total_ratio = (df.iloc[:, 1:-2] != 0.0).sum(axis=1) / 1078
        results = {
            'chid': df['chid'].to_list(),
            'dt': df['dt'].to_list(),
            'query_id': df['query_id'].to_list()
        }
        for shop_cate in self.required_cate:
            cols = [
                "shop_{}_{}_{}".format(shop_cate, self.prefix, dt - 23)
                for dt in range(start_dt, 23)
            ]
            shop_ratio = df[cols][df[cols] != 0].count(axis=1) / 22
            results[shop_cate] = shop_ratio / total_ratio

        feats = pd.DataFrame(results)
        feats = feats.melt(id_vars=['chid', 'dt', 'query_id'],
                           var_name='shop_tag',
                           value_name="relative_freq_buy_{}".format(
                               self.fetch_type))
        return feats

    def get_freq_value(self, df, start_dt=1):
        total_ratio = df.iloc[:, 1:-2][df.iloc[:, 1:-2] != 0.0].mean(axis=1)
        results = {
            'chid': df['chid'].to_list(),
            'dt': df['dt'].to_list(),
            'query_id': df['query_id'].to_list()
        }
        for shop_cate in self.required_cate:
            cols = [
                "shop_{}_{}_{}".format(shop_cate, self.prefix, dt - 23)
                for dt in range(start_dt, 23)
            ]
            shop_ratio = df[cols][df[cols] != 0].mean(axis=1)
            results[shop_cate] = shop_ratio / total_ratio
        feats = pd.DataFrame(results)
        feats = feats.melt(id_vars=['chid', 'dt', 'query_id'],
                           var_name='shop_tag',
                           value_name='relative_{}_buy_{}'.format(
                               self.prefix, self.fetch_type))
        return feats

    def fit_transform_single(self, func, start_dt=1):
        feats_23 = func(self.train_23, start_dt)
        feats_24 = func(self.train_24, start_dt)
        feats_test = func(self.test, start_dt)
        feats_train = pd.concat([
            feats_23.drop(['chid', 'dt'], axis=1),
            feats_24.drop(['chid', 'dt'], axis=1)
        ])
        feats_test = feats_test.drop(['chid', 'dt'], axis=1)
        return feats_train, feats_test
