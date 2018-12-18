import numpy as np
from scipy.stats import gmean, trim_mean
import pandas as pd
import matplotlib.pyplot as plt

class stats:
    def __init__(self,data):
        self.data = data
        self.min = self.data.min()
        self.max = self.data.max()
        self.range = self.data.max() - self.data.min()

    def info(self):
        return self.data.dtypes

    def desc_stats(self):
        results = np.ndarray(shape=(self.data.shape[1], 19), dtype=float, order='F')
        results[:, 0] = self.data.count()
        results[:, 1] = self.data.nunique()
        results[:, 2] = self.data.sum()
        results[:, 3] = self.data.mean()
        results[:, 4] = self.data.median()
        results[:, 5] = self.data.mode().mean()
        results[:, 6] = self.min
        results[:, 7] = self.data.quantile(0.25)
        results[:, 8] = self.data.quantile(0.50)
        results[:, 9] = self.data.quantile(0.75)
        results[:, 10] = self.max
        results[:, 11] = self.range
        results[:, 12] = self.data.std()
        results[:, 13] = self.data.var()
        results[:, 14] = self.data.sem()
        results[:, 15] = np.transpose(gmean(self.data))
        results[:, 16] = np.transpose(trim_mean(self.data, 0.1))
        results[:, 17] = self.data.skew()
        results[:, 18] = self.data.kurtosis()

        Columns = ['Count', 'Distinct', 'Sum', 'Mean', 'Median', 'Mode', 'Min', '%25', '%50', '%75', 'Max', 'Range', 'Std',
               'Var', 'S.E. Mean', 'Gmean', 'Trim Mean', 'Skewness', 'Kurtosis']
        stats = pd.DataFrame(results, columns=Columns, index= self.data.columns.values)
    #norm = np.ndarray(shape=(data.shape[1], 2), dtype=float, order='F')
    #for i in range(0,data.shape[1]):
    #   norm[i] = shapiro(data[i])
    #norm = pd.DataFrame(norm, columns=['Statistic', 'P-value']).round(4)
        return stats

    def interval(self,col,sınıf):
        k = round(self.range/sınıf)
        for i in range(int(self.min[0]),int(self.max[0])):
            cf = pd.value_counts(self.data.ix[i:i+k, col]).to_frame().reset_index()
            df = pd.DataFrame([i, i+k,cf], columns=['start','end','frequency'])
            i += k
        return df

    def frequency(self,col):
        df = pd.value_counts(self.data.ix[:,col]).to_frame().reset_index()
        df.sort_values(by='index')
        return df

    def descriptive_graph(stats,col):
        plt.hist(stats[col])
        plt.ylabel('Probability')
        plt.title('Histogram of ' + stats.columns[col])
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.show()