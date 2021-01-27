```python
# Install a conda package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy
!{sys.executable} -m pip install pandas
!{sys.executable} -m pip install scikit-learn
!{sys.executable} -m pip install matplotlib
!{sys.executable} -m pip install seaborn
```

    Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (1.19.5)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: pandas in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (1.2.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: pytz>=2017.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (2020.5)
    Requirement already satisfied: numpy>=1.16.5 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (1.19.5)
    Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: scikit-learn in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (0.24.1)
    Requirement already satisfied: joblib>=0.11 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.0.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.6.0)
    Requirement already satisfied: numpy>=1.13.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.19.5)
    Requirement already satisfied: numpy>=1.13.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.19.5)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: matplotlib in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (3.3.3)
    Requirement already satisfied: numpy>=1.15 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (1.19.5)
    Requirement already satisfied: pillow>=6.2.0 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (8.1.0)
    Requirement already satisfied: cycler>=0.10 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: python-dateutil>=2.1 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: six in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Requirement already satisfied: six in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 -m pip install --upgrade pip' command.[0m



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
```


```python
cumulative_vaccinations = [
    1959151,
    2080280,
    2254556,
    2494371,
    2769164,
    3090058,
    3365492,
    3520056,
    3687206,
    3985579,
    4303730,
    4661293,
    5085771,
    5727693,
    5962544,
    ]

daily_inoculations = [100000]

for i, x in enumerate(cumulative_vaccinations):
    if len(cumulative_vaccinations) > i+1:
        daily_inoculations.append(cumulative_vaccinations[i+1] - x)

dataframe = pd.DataFrame({'date': pd.date_range(start='1/11/2021', end='1/25/2021'), 'daily': daily_inoculations, 'cumulative': cumulative_vaccinations})
dataframe.set_index('date', inplace=True)
```


```python
from datetime import datetime, timedelta

reg = LinearRegression()

x = np.array([x.timetuple().tm_yday for x in dataframe.index]).reshape(-1, 1)
y = dataframe['daily']

reg.fit(x, y)
```




    LinearRegression()




```python
import math
start = 11 + len(dataframe)
prediction_range = range(start, start + 40)
prediction_range_array = np.array(prediction_range).reshape(-1, 1)
projection = reg.predict(prediction_range_array)
projection = [x - (math.log(x) * 1200 * i) for i, x in enumerate(projection, start=1)]
print(projection)
```

    [422755.33581809513, 427654.3052563802, 432450.32566678204, 437149.841700235, 441758.6069469874, 446281.7842700146, 450724.02817278286, 455089.55297211726, 459382.189648599, 463605.4335861669, 467762.48492081586, 471856.28284820716, 475889.53495865886, 479864.7424520525, 483784.2219180005, 487650.12423606415, 491464.4510480944, 495229.06917334103, 498945.7232719928, 502616.0470106041, 506241.572940666, 509823.74126728537, 513363.90765688434, 516863.35020979674, 520323.27570460964, 523744.82520530984, 527129.0791091335, 530477.061702013, 533789.7452792486, 537068.0538812269, 540312.866687398, 543525.0211060917, 546705.3155929688, 549854.512226797, 552973.3390677144, 556062.4923201238, 559122.6383197288, 562154.4153619654, 565158.4353871128, 568135.2855356494]



```python

prediction_dates = [(datetime(2021, 1, 1) + timedelta(x)) for x in prediction_range]
result_df = pd.DataFrame({'date': prediction_dates, 'daily': projection})
result_df.set_index('date', inplace=True)
```


```python
last_known = dataframe['cumulative'][-1]
# print(last_known)
projection_cumulative = []
for x in projection:
    projection_cumulative.append(last_known + x)
    last_known = last_known + x
result_df['cumulative'] = projection_cumulative
print(result_df)
```

                        daily    cumulative
    date                                   
    2021-01-27  422755.335818  6.385299e+06
    2021-01-28  427654.305256  6.812954e+06
    2021-01-29  432450.325667  7.245404e+06
    2021-01-30  437149.841700  7.682554e+06
    2021-01-31  441758.606947  8.124312e+06
    2021-02-01  446281.784270  8.570594e+06
    2021-02-02  450724.028173  9.021318e+06
    2021-02-03  455089.552972  9.476408e+06
    2021-02-04  459382.189649  9.935790e+06
    2021-02-05  463605.433586  1.039940e+07
    2021-02-06  467762.484921  1.086716e+07
    2021-02-07  471856.282848  1.133901e+07
    2021-02-08  475889.534959  1.181490e+07
    2021-02-09  479864.742452  1.229477e+07
    2021-02-10  483784.221918  1.277855e+07
    2021-02-11  487650.124236  1.326620e+07
    2021-02-12  491464.451048  1.375767e+07
    2021-02-13  495229.069173  1.425290e+07
    2021-02-14  498945.723272  1.475184e+07
    2021-02-15  502616.047011  1.525446e+07
    2021-02-16  506241.572941  1.576070e+07
    2021-02-17  509823.741267  1.627052e+07
    2021-02-18  513363.907657  1.678389e+07
    2021-02-19  516863.350210  1.730075e+07
    2021-02-20  520323.275705  1.782107e+07
    2021-02-21  523744.825205  1.834482e+07
    2021-02-22  527129.079109  1.887195e+07
    2021-02-23  530477.061702  1.940242e+07
    2021-02-24  533789.745279  1.993621e+07
    2021-02-25  537068.053881  2.047328e+07
    2021-02-26  540312.866687  2.101360e+07
    2021-02-27  543525.021106  2.155712e+07
    2021-02-28  546705.315593  2.210383e+07
    2021-03-01  549854.512227  2.265368e+07
    2021-03-02  552973.339068  2.320665e+07
    2021-03-03  556062.492320  2.376272e+07
    2021-03-04  559122.638320  2.432184e+07
    2021-03-05  562154.415362  2.488399e+07
    2021-03-06  565158.435387  2.544915e+07
    2021-03-07  568135.285536  2.601729e+07



```python
complete = pd.concat([dataframe, result_df])
print(complete)

```

                       daily    cumulative
    date                                  
    2021-01-11  1.000000e+05  1.959151e+06
    2021-01-12  1.211290e+05  2.080280e+06
    2021-01-13  1.742760e+05  2.254556e+06
    2021-01-14  2.398150e+05  2.494371e+06
    2021-01-15  2.747930e+05  2.769164e+06
    2021-01-16  3.208940e+05  3.090058e+06
    2021-01-17  2.754340e+05  3.365492e+06
    2021-01-18  1.545640e+05  3.520056e+06
    2021-01-19  1.671500e+05  3.687206e+06
    2021-01-20  2.983730e+05  3.985579e+06
    2021-01-21  3.181510e+05  4.303730e+06
    2021-01-22  3.575630e+05  4.661293e+06
    2021-01-23  4.244780e+05  5.085771e+06
    2021-01-24  6.419220e+05  5.727693e+06
    2021-01-25  2.348510e+05  5.962544e+06
    2021-01-27  4.383442e+05  6.400888e+06
    2021-01-28  4.589423e+05  6.859831e+06
    2021-01-29  4.795404e+05  7.339371e+06
    2021-01-30  5.001385e+05  7.839510e+06
    2021-01-31  5.207366e+05  8.360246e+06
    2021-02-01  5.413347e+05  8.901581e+06
    2021-02-02  5.619328e+05  9.463514e+06
    2021-02-03  5.825309e+05  1.004604e+07
    2021-02-04  6.031290e+05  1.064917e+07
    2021-02-05  6.237271e+05  1.127290e+07
    2021-02-06  6.443251e+05  1.191723e+07
    2021-02-07  6.649232e+05  1.258215e+07
    2021-02-08  6.855213e+05  1.326767e+07
    2021-02-09  7.061194e+05  1.397379e+07
    2021-02-10  7.267175e+05  1.470051e+07
    2021-02-11  7.473156e+05  1.544782e+07
    2021-02-12  7.679137e+05  1.621574e+07
    2021-02-13  7.885118e+05  1.700425e+07
    2021-02-14  8.091099e+05  1.781336e+07
    2021-02-15  8.297079e+05  1.864307e+07
    2021-02-16  8.503060e+05  1.949337e+07
    2021-02-17  8.709041e+05  2.036428e+07
    2021-02-18  8.915022e+05  2.125578e+07
    2021-02-19  9.121003e+05  2.216788e+07
    2021-02-20  9.326984e+05  2.310058e+07
    2021-02-21  9.532965e+05  2.405387e+07
    2021-02-22  9.738946e+05  2.502777e+07
    2021-02-23  9.944927e+05  2.602226e+07
    2021-02-24  1.015091e+06  2.703735e+07
    2021-02-25  1.035689e+06  2.807304e+07
    2021-02-26  1.056287e+06  2.912933e+07
    2021-02-27  1.076885e+06  3.020621e+07
    2021-02-28  1.097483e+06  3.130370e+07
    2021-03-01  1.118081e+06  3.242178e+07
    2021-03-02  1.138679e+06  3.356046e+07



```python
fig = plt.figure()
ax = complete.drop('cumulative', axis=1).plot()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig.show()

fig = plt.figure()
ax = complete.drop('daily', axis=1).plot()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig.show()
```


    <Figure size 432x288 with 0 Axes>



    
![svg](vaccinations_files/vaccinations_8_1.svg)
    



    <Figure size 432x288 with 0 Axes>



    
![svg](vaccinations_files/vaccinations_8_3.svg)
    



```python
import seaborn as sns
sns.set_theme(style="darkgrid")
g = sns.relplot(kind="line", data=dataframe)
g.fig.autofmt_xdate()

```


    
![svg](vaccinations_files/vaccinations_9_0.svg)
    



```python

```
