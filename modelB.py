import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from numpy import dot
from numpy.linalg import norm


class BModel:
    def __init__(self):
        self.sc = StandardScaler()

    def make_sc(self, input_df):
        self.sc.fit(input_df.drop('food', axis=1))

    def pred(self, log_df: pd.DataFrame, situation: pd.DataFrame):
        labels = log_df['food']
        log_ndarray = self.sc.transform(log_df.drop('food', axis=1))
        input_situation = self.sc.transform(situation)[0]

        reco_dict = {}
        for situation_log, label in zip(log_ndarray, labels):
            cos_sim = dot(situation_log, input_situation) / (norm(situation_log) * norm(input_situation))
            reco_dict[cos_sim] = label

        sort_list = sorted(reco_dict.items(), reverse=True)

        reco_list = []
        for x in sort_list:
            if len(reco_list) >= 3:
                break

            if x[1] not in reco_list:
                reco_list.append(x[1])

        return reco_list


if __name__ == '__main__':
    path = 'test.csv'
    _df = pd.read_csv(path)

    l = [[2, 35, 62]]
    sample = pd.DataFrame(l, columns=['time', 'temperature', 'humidity'])

    b_model = BModel()
    b_model.make_sc(_df)
    _reco_list = b_model.pred(_df, sample)
    print(_reco_list)
