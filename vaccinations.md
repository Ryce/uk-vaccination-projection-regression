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
    6221850,
    6473752,
    6816945,
    ]

daily_inoculations = [100000]

for i, x in enumerate(cumulative_vaccinations):
    if len(cumulative_vaccinations) > i+1:
        daily_inoculations.append(cumulative_vaccinations[i+1] - x)

dataframe = pd.DataFrame({'date': pd.date_range(start='1/11/2021', end='1/28/2021'), 'daily': daily_inoculations, 'cumulative': cumulative_vaccinations})
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
prediction_range = range(start, start + 50)
prediction_range_array = np.array(prediction_range).reshape(-1, 1)
projection = reg.predict(prediction_range_array)
projection = [x - (math.log(x) * 900 * i) for i, x in enumerate(projection, start=1)]
print(projection)
```

    [381055.1779837059, 382868.6167675855, 384624.3260742873, 386325.02612076094, 387973.21558427915, 389571.1962509129, 391121.0942528703, 392624.8784525218, 394084.3764266035, 395501.28842157614, 396877.19958540413, 398213.5907283472, 399511.8478228629, 400773.2704182546, 401999.0791175689, 403190.42224118684, 404348.3817825391, 405473.97874563077, 406568.17794096365, 407631.8923054968, 408665.9868031052, 409671.2819542622, 410648.5570371286, 411598.5529966863, 412521.9750938249, 413419.495322257, 414291.7546176738, 415139.3648805831, 415962.91083170165, 416762.9517165629, 417540.02287407167, 418294.6371820721, 419027.2863915348, 419738.44235969766, 420428.55819138105, 421098.0692967188, 421747.3943726877, 422376.93631505675, 422987.0830667118, 423578.2084077132, 424150.6726919234, 424704.82353456697, 425240.99645467976, 425759.5154760217, 426260.69368970534, 426744.8337814907, 427212.2285264337, 427663.16125333577, 428097.9062812285, 428516.72932993446]



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
    2021-01-29  381055.177984  6.854807e+06
    2021-01-30  382868.616768  7.237676e+06
    2021-01-31  384624.326074  7.622300e+06
    2021-02-01  386325.026121  8.008625e+06
    2021-02-02  387973.215584  8.396598e+06
    2021-02-03  389571.196251  8.786170e+06
    2021-02-04  391121.094253  9.177291e+06
    2021-02-05  392624.878453  9.569916e+06
    2021-02-06  394084.376427  9.964000e+06
    2021-02-07  395501.288422  1.035950e+07
    2021-02-08  396877.199585  1.075638e+07
    2021-02-09  398213.590728  1.115459e+07
    2021-02-10  399511.847823  1.155410e+07
    2021-02-11  400773.270418  1.195488e+07
    2021-02-12  401999.079118  1.235688e+07
    2021-02-13  403190.422241  1.276007e+07
    2021-02-14  404348.381783  1.316441e+07
    2021-02-15  405473.978746  1.356989e+07
    2021-02-16  406568.177941  1.397646e+07
    2021-02-17  407631.892305  1.438409e+07
    2021-02-18  408665.986803  1.479276e+07
    2021-02-19  409671.281954  1.520243e+07
    2021-02-20  410648.557037  1.561307e+07
    2021-02-21  411598.552997  1.602467e+07
    2021-02-22  412521.975094  1.643720e+07
    2021-02-23  413419.495322  1.685061e+07
    2021-02-24  414291.754618  1.726491e+07
    2021-02-25  415139.364881  1.768005e+07
    2021-02-26  415962.910832  1.809601e+07
    2021-02-27  416762.951717  1.851277e+07
    2021-02-28  417540.022874  1.893031e+07
    2021-03-01  418294.637182  1.934861e+07
    2021-03-02  419027.286392  1.976763e+07
    2021-03-03  419738.442360  2.018737e+07
    2021-03-04  420428.558191  2.060780e+07
    2021-03-05  421098.069297  2.102890e+07
    2021-03-06  421747.394373  2.145065e+07
    2021-03-07  422376.936315  2.187302e+07
    2021-03-08  422987.083067  2.229601e+07
    2021-03-09  423578.208408  2.271959e+07
    2021-03-10  424150.672692  2.314374e+07
    2021-03-11  424704.823535  2.356844e+07
    2021-03-12  425240.996455  2.399368e+07
    2021-03-13  425759.515476  2.441944e+07
    2021-03-14  426260.693690  2.484571e+07
    2021-03-15  426744.833781  2.527245e+07
    2021-03-16  427212.228526  2.569966e+07
    2021-03-17  427663.161253  2.612733e+07
    2021-03-18  428097.906281  2.655542e+07
    2021-03-19  428516.729330  2.698394e+07



```python
complete = pd.concat([dataframe, result_df])
print(complete)

```

                        daily    cumulative
    date                                   
    2021-01-11  100000.000000  1.959151e+06
    2021-01-12  121129.000000  2.080280e+06
    2021-01-13  174276.000000  2.254556e+06
    2021-01-14  239815.000000  2.494371e+06
    2021-01-15  274793.000000  2.769164e+06
    ...                   ...           ...
    2021-03-15  426744.833781  2.527245e+07
    2021-03-16  427212.228526  2.569966e+07
    2021-03-17  427663.161253  2.612733e+07
    2021-03-18  428097.906281  2.655542e+07
    2021-03-19  428516.729330  2.698394e+07
    
    [67 rows x 2 columns]



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
    

