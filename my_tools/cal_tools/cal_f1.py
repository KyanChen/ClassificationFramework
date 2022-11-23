import numpy as np

#data = '0.6740    0.6730    0.6703 '
# data = data.split()
# data = [float(x) for x in data]
# f1 = 2 * data[0] * data[1] / (data[0] + data[1])
# print(f1)
# print(data[-1])
t = \
'''
1 industrial land     0.6852    0.7400    0.7115       200
          10 shrub land     0.7708    0.7400    0.7551       200
   11 natural grassland     0.8762    0.9200    0.8976       200
12 artificial grassland     0.8476    0.6950    0.7637       200
               13 river     0.5891    0.5950    0.5920       200
                14 lake     0.6295    0.7900    0.7007       200
                15 pond     0.6339    0.3550    0.4551       200
    2 urban residential     0.6250    0.6750    0.6490       200
    3 rural residential     0.7421    0.7050    0.7231       200
         4 traffic land     0.7075    0.7500    0.7282       200
          5 paddy field     0.6781    0.7900    0.7298       200
       6 irrigated land     0.4089    0.4600    0.4329       200
         7 dry cropland     0.7838    0.7250    0.7532       200
          8 garden plot     0.7407    0.5000    0.5970       200
       9 arbor woodland     0.6732    0.8650    0.7571       200

               accuracy                         0.6870      3000
              macro avg     0.6928    0.6870    0.6831      3000
           weighted avg     0.6928    0.6870    0.6831      3000'''
t = t.split('\n')
t = [x for x in t if len(x) >2]
t = [x.split() for x in t]
t = [x[-4:-1] for x in t]
t = t[:15]
t = [[float(m) for m in x] for x in t]
t = np.array(t)
t = np.mean(t, axis=0)
print(2*t[0]*t[1]/(t[0]+t[1]))
print(t)