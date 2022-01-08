import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import re
import sys

sys.path.insert(0, './code')
import dataloader  # noqa: E402


class AmtTrainHandler():
    def __init__(self):
        self.ylabels = [
            "2", "6", "10", '12', "13", "15", "18", "19", "21", "22", "25",
            "26", "36", "37", "39", "48"
        ]
        self.ylabel_cols_24 = [
            "shop_{}_amt_24".format(ylabel) for ylabel in self.ylabels
        ]
        self.ylabel_cols_23 = [
            "shop_{}_amt_23".format(ylabel) for ylabel in self.ylabels
        ]
        self.shop_cate = [str(i + 1) for i in range(48)] + ['other']
        self.cols_24 = [
            "shop_{}_amt_24".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_23 = [
            "shop_{}_amt_23".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_1 = [
            "shop_{}_amt_1".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_2 = [
            "shop_{}_amt_2".format(a_cate) for a_cate in self.shop_cate
        ]

    def update_data(self, data):
        print("Start Update Data")
        self.data = data.copy().reset_index(drop=True)
        self.get_train_test()
        print("Finished updating data")
        del self.data

    def get_new_cols(self, df, dt):
        reg = r"(.+amt_)\d+"
        n_cols = ['chid']
        for col in df.drop('chid', axis=1).columns:
            n_idx = int(col.split('_')[-1]) - dt
            n_col = re.findall(reg, col)[0]
            n_col = n_col + str(n_idx)
            n_cols.append(n_col)
        return n_cols

    def get_train_test(self):
        # train set 1~23 pred 24
        # test set 2~24
        # label for train set
        self.y_24 = self.data[self.ylabel_cols_24].copy()
        self.y_23 = self.data[self.ylabel_cols_23].copy()
        self.y_24.columns = self.ylabels
        self.y_23.columns = self.ylabels
        self.y = pd.concat([self.y_23, self.y_24])
        self.y_total_24 = self.data[self.cols_24].copy()
        self.y_total_23 = self.data[self.cols_23].copy()

        # X
        self.X_23 = self.data.drop(self.cols_23 + self.cols_24, axis=1)
        self.X_24 = self.data.drop(self.cols_1 + self.cols_24, axis=1).copy()
        n_cols_23 = self.get_new_cols(self.X_23, 23)
        n_cols_24 = self.get_new_cols(self.X_24, 24)
        self.X_23.columns = n_cols_23
        self.X_24.columns = n_cols_24
        self.X = pd.concat([self.X_23, self.X_24])
        # test set
        self.test = self.data.drop(self.cols_1 + self.cols_2, axis=1).copy()
        n_cols_25 = self.get_new_cols(self.test, 25)
        self.test.columns = n_cols_25
        # self.test = self.test.drop('chid', axis=1)

    def fit(self):
        X = self.X.drop('chid', axis=1)
        y = self.y
        kf = KFold(n_splits=3, shuffle=True, random_state=16)
        kf.get_n_splits(X)
        return X, y, kf


class AmtProfileHandler():
    def __init__(self):
        self.cate_feats = [
            'masts',
            'educd',
            'trdtp',
            'naty',
            'poscd',
            'cuorg',
            'primary_card',
            'age',
            'gender_code',
            'card_1',
            'card_1_12',
            'card_1_16',
            'card_2',
            'card_2_12',
            'card_2_16',
        ]
        self.str_cate = [
            'card_1', 'card_2', 'card_1_12', 'card_1_16', 'card_2_12',
            'card_2_16'
        ]
        self.label_encoder = {}

    def label_encoding(self):
        for cate_feat in self.cate_feats:
            le = LabelEncoder()
            if cate_feat not in self.str_cate:
                self.data[cate_feat] = self.data[cate_feat].apply(
                    lambda x: int(x))
            else:
                self.data[cate_feat] = self.data[cate_feat].apply(
                    lambda x: str(x))
            le.fit(self.data[cate_feat])
            self.label_encoder.update({cate_feat: deepcopy(le)})
            self.data[cate_feat] = le.transform(self.data[cate_feat])

    def update_data(self, data):
        print("Start Update Data")
        self.data = data.copy()
        self.data = self.data.fillna(-1)
        print("Finished updating data")
        print("start label encoding")
        self.label_encoding()
        print('Finish labor encoding')
        del self.data

    def transform(self, df):
        df = df.fillna(-1)
        for cate_feat in self.cate_feats:
            if cate_feat not in self.str_cate:
                df[cate_feat] = df[cate_feat].apply(lambda x: int(x))
            else:
                df[cate_feat] = df[cate_feat].apply(lambda x: str(x))
            df[cate_feat] = self.label_encoder[cate_feat].transform(
                df[cate_feat])
        for feat in self.cate_feats:
            df[feat] = df[feat].fillna(-1)
        return df


class CntTrainHandler():
    def __init__(self):
        self.ylabels = [
            "2", "6", "10", '12', "13", "15", "18", "19", "21", "22", "25",
            "26", "36", "37", "39", "48"
        ]
        self.ylabel_cols_24 = [
            "shop_{}_cnt_24".format(ylabel) for ylabel in self.ylabels
        ]
        self.ylabel_cols_23 = [
            "shop_{}_cnt_23".format(ylabel) for ylabel in self.ylabels
        ]
        self.shop_cate = [str(i + 1) for i in range(48)] + ['other']
        self.cols_24 = [
            "shop_{}_cnt_24".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_23 = [
            "shop_{}_cnt_23".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_1 = [
            "shop_{}_cnt_1".format(a_cate) for a_cate in self.shop_cate
        ]
        self.cols_2 = [
            "shop_{}_cnt_2".format(a_cate) for a_cate in self.shop_cate
        ]

    def update_data(self, data):
        print("Start Update Data")
        self.data = data.copy()
        self.get_train_test()
        print("Finished updating data")
        del self.data

    def get_new_cols(self, df, dt):
        n_cols = ['chid']
        reg = r"(.+cnt_)\d+"
        for col in df.drop('chid', axis=1).columns:
            n_idx = int(col.split('_')[-1]) - dt
            n_col = re.findall(reg, col)[0]
            n_col = n_col + str(n_idx)
            n_cols.append(n_col)
        return n_cols

    def get_label_value(self):
        self.y_24 = self.data[self.ylabel_cols_24].copy()
        self.y_23 = self.data[self.ylabel_cols_23].copy()
        self.y_24.columns = self.ylabels
        self.y_23.columns = self.ylabels
        self.y = pd.concat([self.y_23, self.y_24])
        # for col in self.y.columns:
        #     self.y[col] = self.y[col].apply(lambda x: 1 if x > 0 else 0)

    def get_train_test(self):
        # ylabel
        print("Start Processing y label")
        self.get_label_value()
        # train set
        print("Start Processing train set")
        self.train_23 = self.data.drop(self.cols_23 + self.cols_24,
                                       axis=1).copy()
        self.train_24 = self.data.drop(self.cols_1 + self.cols_24,
                                       axis=1).copy()
        n_cols_23 = self.get_new_cols(self.train_23, 23)
        n_cols_24 = self.get_new_cols(self.train_24, 24)
        self.train_23.columns = n_cols_23
        self.train_24.columns = n_cols_24
        self.train = pd.concat([self.train_23, self.train_24])
        # test set
        print("Start Processing test set")
        self.test = self.data.drop(self.cols_1 + self.cols_2, axis=1).copy()
        n_cols_25 = self.get_new_cols(self.test, 25)
        self.test.columns = n_cols_25


class RankTopHandler():
    def __init__(self):
        self.ylabels = [
            "2", "6", "10", '12', "13", "15", "18", "19", "21", "22", "25",
            "26", "36", "37", "39", "48"
        ]

        self.cols_24 = ["top{}_24".format(rank) for rank in range(1, 4)] + [
            "imp_top{}_24".format(rank) for rank in range(1, 4)
        ] + ["how_many_cate_24", "how_many_cate_imp_24"]
        self.cols_23 = ["top{}_23".format(rank) for rank in range(1, 4)] + [
            "imp_top{}_23".format(rank) for rank in range(1, 4)
        ] + ["how_many_cate_23", "how_many_cate_imp_23"]
        self.cols_1 = ["top{}_1".format(rank) for rank in range(1, 4)] + [
            "imp_top{}_1".format(rank) for rank in range(1, 4)
        ] + ["how_many_cate_1", "how_many_cate_imp_1"]
        self.cols_2 = ["top{}_2".format(rank) for rank in range(1, 4)] + [
            "imp_top{}_2".format(rank) for rank in range(1, 4)
        ] + ["how_many_cate_2", "how_many_cate_imp_2"]

    def update_data(self, data):
        print("Start Update Data")
        self.data = data.copy()
        self.get_train_test()
        print("Finished updating data")
        del self.data

    def get_new_cols(self, df, dt):
        n_cols = ['chid']
        reg = r"(.+_)\d+"
        for col in df.drop('chid', axis=1).columns:
            n_idx = int(col.split('_')[-1]) - dt
            n_col = re.findall(reg, col)[0]
            n_col = n_col + str(n_idx)
            n_cols.append(n_col)
        return n_cols

    def get_train_test(self):
        # train set
        print("Start Processing train set")
        self.train_23 = self.data.drop(self.cols_23 + self.cols_24,
                                       axis=1).copy()
        self.train_24 = self.data.drop(self.cols_1 + self.cols_24,
                                       axis=1).copy()

        n_cols_23 = self.get_new_cols(self.train_23, 23)
        n_cols_24 = self.get_new_cols(self.train_24, 24)
        self.train_23.columns = n_cols_23
        self.train_24.columns = n_cols_24
        self.train = pd.concat([self.train_23, self.train_24])

        # test set
        print("Start Processing test set")
        self.test = self.data.drop(self.cols_1 + self.cols_2, axis=1).copy()
        n_cols_25 = self.get_new_cols(self.test, 25)
        self.test.columns = n_cols_25


class RegionTrainHandler():
    def __init__(self):
        self.cols = [
            "domestic_offline_cnt",
            "domestic_online_cnt",
            "overseas_offline_cnt",
            "overseas_online_cnt",
            "domestic_offline_amt",
            "domestic_online_amt",
            "overseas_offline_amt",
            "overseas_online_amt",
        ]
        self.cols_24 = [col + "_{}".format(24) for col in self.cols]
        self.cols_23 = [col + "_{}".format(23) for col in self.cols]
        self.cols_1 = [col + "_{}".format(1) for col in self.cols]
        self.cols_2 = [col + "_{}".format(2) for col in self.cols]

    def update_data(self, data):
        print("Start Update Data")
        self.data = data.copy()
        self.get_train_test()
        print("Finished Update Data")
        del self.data

    def get_new_cols(self, df, dt):
        n_cols = ['chid']
        reg = r"(.+_)\d+"
        for col in df.drop('chid', axis=1).columns:
            n_idx = int(col.split('_')[-1]) - dt
            n_col = re.findall(reg, col)[0]
            n_col = n_col + str(n_idx)
            n_cols.append(n_col)
        return n_cols

    def get_train_test(self):
        # train set
        print("Start Processing train set")
        self.train_23 = self.data.drop(self.cols_23 + self.cols_24,
                                       axis=1).copy()
        self.train_24 = self.data.drop(self.cols_1 + self.cols_24,
                                       axis=1).copy()
        n_cols_23 = self.get_new_cols(self.train_23, 23)
        n_cols_24 = self.get_new_cols(self.train_24, 24)
        self.train_23.columns = n_cols_23
        self.train_24.columns = n_cols_24
        self.train = pd.concat([self.train_23, self.train_24])

        # test set
        print("Start Processing test set")
        self.test = self.data.drop(self.cols_1 + self.cols_2, axis=1).copy()
        n_cols_25 = self.get_new_cols(self.test, 25)
        self.test.columns = n_cols_25


class StackTrainHandler():
    def __init__(self):
        self.required_cate = [
            "2", "6", "10", '12', "13", "15", "18", "19", "21", "22", "25",
            "26", "36", "37", "39", "48"
        ]
        self.loader = dataloader.DataLoader()
        self.get_stack_config()
        print("Start Loading idx info")
        self.get_idx_results()
        print("Start Loading ylabels info")
        self.get_ylabels()

    def get_feats(self, results):
        r_list = [{}, {}, {}]
        for ylabel in self.required_cate:
            y_result = results[ylabel]
            for i, df in enumerate(y_result):
                r_list[i].update({"{}".format(ylabel): df[0].to_list()})
        r_list = [pd.DataFrame(r_dict) for r_dict in r_list]
        return r_list

    def get_idx_results(self):
        print("Fetch idx_result from {}".format(self.config['idx_results']))
        self.idx_results = self.loader.load_result(self.config['idx_results'])

    def get_ylabels(self):
        ylabel_path = self.config['ylabels']
        print("Fetch ylabel infos from {}".format(ylabel_path))
        self.ylabels, self.test_labels, self.train_labels = self.loader.load_result(  # noqa: E501
            ylabel_path)
        tmp_test = pd.DataFrame({
            'chid': list(self.test_labels['chid'].unique()),
            'dt': [25] * 500000
        })
        tmp_test['query_id'] = tmp_test['chid'].apply(
            lambda x: str(x)) + tmp_test['dt'].apply(lambda x: str(x))
        self.tmp_test = tmp_test

    def get_base_model_feats(self, model_path, model_name):
        test_results, train_results, _ = self.loader.load_result(model_path)
        test_raw_feats = self.get_feats(test_results)
        train_raw_feats = self.get_feats(train_results)

        test_feats = []
        train_feats = []
        for test_feat in test_raw_feats:
            feat = pd.concat([self.tmp_test, test_feat], axis=1)
            feat = feat.melt(id_vars=["chid", 'dt', 'query_id'],
                             var_name="shop_tag",
                             value_name="pred_{}".format(model_name))
            test_feats.append(feat)

        self.config['test_model_feats'].update({model_name: test_feats})

        for i, train_feat in enumerate(train_raw_feats):
            tmp_train = self.train_labels[i].reset_index(drop=True)
            tmp_train['query_id'] = tmp_train['chid'].apply(
                lambda x: str(x)) + tmp_train['dt'].apply(lambda x: str(x))
            feat = pd.concat([tmp_train, train_feat], axis=1)
            feat = feat.melt(id_vars=['chid', 'dt', 'query_id'],
                             var_name="shop_tag",
                             value_name="pred_{}".format(model_name))
            train_feats.append(feat)
        self.config['train_model_feats'].update({model_name: train_feats})

    def merge_base_model_feats(self):
        model_names = self.config['base_model_names']
        print(model_names)
        train_model_feats = [
            self.config['train_model_feats'][model_name]
            for model_name in model_names
        ]
        test_model_feats = [
            self.config['test_model_feats'][model_name]
            for model_name in model_names
        ]
        # test feats
        test_feats = []
        for feats in zip(*test_model_feats):
            f_feat = [feats[0]]
            o_feat = [
                feat[['pred_{}'.format(model_name)]]
                for model_name, feat in zip(model_names[1:], feats[1:])
            ]
            test_feat = pd.concat(f_feat + o_feat, axis=1)
            test_feats.append(test_feat)

        # train feats
        train_feats = []
        for feats in zip(*train_model_feats):
            f_feat = [feats[0]]
            o_feat = [
                feat[['pred_{}'.format(model_name)]]
                for model_name, feat in zip(model_names[1:], feats[1:])
            ]
            train_feat = pd.concat(f_feat + o_feat, axis=1)
            train_feats.append(train_feat)

        self.test_feats = test_feats
        self.train_feats = train_feats

    def get(self):
        model_names = self.config['base_model_names']
        model_paths = self.config['base_model_paths']
        print("Start transform the base model results into ltr feats")
        for model_name, model_path in zip(model_names, model_paths):
            print("Start trans model {} from {}".format(
                model_name, model_path))
            self.get_base_model_feats(model_path, model_name)
        print("Start merge the base model feats")
        self.merge_base_model_feats()

    def get_stack_config(self):
        self.config = {
            "idx_results":
            "2021_12_20_idx_results.joblib",
            "ylabels":
            '2021_12_20_stack_labels.joblib',
            "base_model_paths": [
                '2021_12_20_amt_train_results.joblib',
                '2021_12_20_cnt_train_results.joblib',
                '2021_12_20_amt_cnt_train_results.joblib',
                '2021_12_22_amt_cnt_train_results.joblib',
                '2021_12_24_cnt_v_train_results.joblib',
                "2021_12_27_cnt_poisson_train_results.joblib",
            ],
            "base_model_names": [
                "amt_1",
                "cnt_1",
                "amt_2",
                "cnt_2",
                "cnt_v_1",
                "cnt_poisson_1",
            ],
            "train_model_feats": {},
            "test_model_feats": {},
        }
