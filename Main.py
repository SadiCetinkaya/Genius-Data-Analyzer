from sklearn import datasets
import pandas as pd
from Functions.Descriptive import *

boston = datasets.load_boston()

b = pd.DataFrame(boston.data,columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])

main = stats(b)
info = main.info()
stats = main.desc_stats()
freq = main.frequency(3)
#inter = main.interval(0,5)

cf = pd.value_counts(b.ix[10:15, 1]).to_frame().reset_index()
ef = b.ix[10:13, 1].shape[0]

print(b.head() , "\n")
print(info , "\n")

print(stats , "\n")

print(freq , "\n")

print(ef , "\n")