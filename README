```python
# Install a conda package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install numpy
!{sys.executable} -m pip install pandas
!{sys.executable} -m pip install scikit-learn
!{sys.executable} -m pip install matplotlib
```

    Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (1.19.5)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/usr/local/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: pandas in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (1.2.1)
    Requirement already satisfied: numpy>=1.16.5 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (1.19.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: pytz>=2017.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from pandas) (2020.5)
    Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/usr/local/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: scikit-learn in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (0.24.1)
    Requirement already satisfied: numpy>=1.13.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.19.5)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.0.0)
    Requirement already satisfied: scipy>=0.19.1 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.6.0)
    Requirement already satisfied: numpy>=1.13.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from scikit-learn) (1.19.5)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/usr/local/bin/python3 -m pip install --upgrade pip' command.[0m
    Collecting matplotlib
      Downloading matplotlib-3.3.3-cp39-cp39-macosx_10_9_x86_64.whl (8.5 MB)
    [K     |████████████████████████████████| 8.5 MB 6.9 MB/s 
    [?25hRequirement already satisfied: python-dateutil>=2.1 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: numpy>=1.15 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (1.19.5)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from matplotlib) (2.4.7)
    Collecting cycler>=0.10
      Downloading cycler-0.10.0-py2.py3-none-any.whl (6.5 kB)
    Requirement already satisfied: six in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Collecting kiwisolver>=1.0.1
      Downloading kiwisolver-1.3.1-cp39-cp39-macosx_10_9_x86_64.whl (61 kB)
    [K     |████████████████████████████████| 61 kB 520 kB/s 
    [?25hCollecting pillow>=6.2.0
      Downloading Pillow-8.1.0-cp39-cp39-macosx_10_10_x86_64.whl (2.2 MB)
    [K     |████████████████████████████████| 2.2 MB 1.2 MB/s 
    [?25hRequirement already satisfied: six in /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Installing collected packages: pillow, kiwisolver, cycler, matplotlib
    Successfully installed cycler-0.10.0 kiwisolver-1.3.1 matplotlib-3.3.3 pillow-8.1.0
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/usr/local/bin/python3 -m pip install --upgrade pip' command.[0m



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    ]

daily_inoculations = []

for i, x in enumerate(cumulative_vaccinations):
    if len(cumulative_vaccinations) > i+1:
        daily_inoculations.append(cumulative_vaccinations[i+1] - x)

daily_inoculations = pd.DataFrame({'date': pd.date_range(start='1/11/2021', end='1/21/2021'), 'delta': daily_inoculations})
daily_inoculations.set_index('date', inplace=True)
```


```python
print(daily_inoculations.index)
```

    DatetimeIndex(['2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14',
                   '2021-01-15', '2021-01-16', '2021-01-17', '2021-01-18',
                   '2021-01-19', '2021-01-20', '2021-01-21'],
                  dtype='datetime64[ns]', name='date', freq=None)



```python
from datetime import datetime, timedelta
regressor = LinearRegression()
xaxis = np.array([x.timetuple().tm_yday for x in daily_inoculations.index]).reshape(-1, 1)
regressor.fit(X=xaxis, y=daily_inoculations['delta'])
```




    LinearRegression()




```python
prediction_range = range(len(daily_inoculations), 100)
prediction_range_array = np.array(prediction_range).reshape(-1, 1)
projection = regressor.predict(prediction_range_array)
print(projection)
```

    [ 175116.18181818  189222.8         203329.41818182  217436.03636364
      231542.65454545  245649.27272727  259755.89090909  273862.50909091
      287969.12727273  302075.74545455  316182.36363636  330288.98181818
      344395.6         358502.21818182  372608.83636364  386715.45454545
      400822.07272727  414928.69090909  429035.30909091  443141.92727273
      457248.54545455  471355.16363636  485461.78181818  499568.4
      513675.01818182  527781.63636364  541888.25454545  555994.87272727
      570101.49090909  584208.10909091  598314.72727273  612421.34545455
      626527.96363636  640634.58181818  654741.2         668847.81818182
      682954.43636364  697061.05454545  711167.67272727  725274.29090909
      739380.90909091  753487.52727273  767594.14545455  781700.76363636
      795807.38181818  809914.          824020.61818182  838127.23636364
      852233.85454545  866340.47272727  880447.09090909  894553.70909091
      908660.32727273  922766.94545455  936873.56363636  950980.18181818
      965086.8         979193.41818182  993300.03636364 1007406.65454545
     1021513.27272727 1035619.89090909 1049726.50909091 1063833.12727273
     1077939.74545455 1092046.36363636 1106152.98181818 1120259.6
     1134366.21818182 1148472.83636364 1162579.45454545 1176686.07272727
     1190792.69090909 1204899.30909091 1219005.92727273 1233112.54545455
     1247219.16363636 1261325.78181818 1275432.4        1289539.01818182
     1303645.63636364 1317752.25454545 1331858.87272727 1345965.49090909
     1360072.10909091 1374178.72727273 1388285.34545455 1402391.96363636
     1416498.58181818]



```python

prediction_dates = [(datetime(2021, 1, 1) + timedelta(x)) for x in prediction_range]
result_df = pd.DataFrame({'date': prediction_dates, 'delta': projection})
result_df.set_index('date', inplace=True)
print(result_df)
```

                       delta
    date                    
    2021-01-12  1.751162e+05
    2021-01-13  1.892228e+05
    2021-01-14  2.033294e+05
    2021-01-15  2.174360e+05
    2021-01-16  2.315427e+05
    ...                  ...
    2021-04-06  1.360072e+06
    2021-04-07  1.374179e+06
    2021-04-08  1.388285e+06
    2021-04-09  1.402392e+06
    2021-04-10  1.416499e+06
    
    [89 rows x 1 columns]



```python
complete = pd.concat([daily_inoculations, result_df])
print(complete)
```


```python
fig = plt.figure()
ax = complete.plot()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

fig.show()
```


    <Figure size 432x288 with 0 Axes>



    
![svg](vaccinations_files/vaccinations_8_1.svg)
    



```python

```
