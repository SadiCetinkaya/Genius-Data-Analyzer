from sklearn import datasets
import pandas as pd
from Functions.Descriptive import *

if __name__ == "__main__":
    boston = datasets.load_boston()
    b = pd.DataFrame(boston.data,columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
                              'LSTAT'])
    main = stats(b)
    info = main.info()
    stats = main.desc_stats()
    freq = main.frequency(2)
    #inter = main.interval(1,10)
    kategori = main.variables(0)
    sayısal = main.variables(1)
    dummy = main.variables(2)

    ef = b[(b.ix[:,1]>50) & (b.ix[:,1]<100)].shape[0]

    print(b.head() , "\n")
    print(info , "\n")
    print(stats , "\n")
    print(freq , "\n")
    print(ef, "\n")
    print("kategori: ", kategori, "\n")
    print("sayısal: ",sayısal, "\n")
    print("bool: ", dummy, "\n")