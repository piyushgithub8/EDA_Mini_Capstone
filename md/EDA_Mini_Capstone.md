# If Client Will Subscribe A Term Deposit Or Not

## Introduction
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
### Product Description
A term deposit is a type of deposit account held at a financial institution where money is locked up for some set period of time. Term deposits are usually short-term deposits with maturities ranging from one month to a few years.
### Dataset Description
<ol>
    <li>age: (numeric)
<li>job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
<li>marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
<li>education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
<li>default: has credit in default? (categorical: 'no','yes','unknown')
<li>balance: amount in customer's bank account
<li>housing: has housing loan? (categorical: 'no','yes','unknown')
<li>loan: has personal loan? (categorical: 'no','yes','unknown')
<li>contact: contact communication type (categorical: 'cellular','telephone')
<li>month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
<li>day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
<li>duration: last contact duration, in seconds (numeric). 
<li>campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
<li>pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
<li>previous: number of contacts performed before this campaign and for this client (numeric)
<li>poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
<li>y - has the client subscribed a term deposit? (binary: 'yes','no') </ol>

### Goal
The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).


### Importing Libraries


```python
#importing modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Reading Dataset


```python
train=pd.read_csv(r"C:\Users\asus\Downloads\train.csv",sep=';') #reading train dataset
```


```python
test=pd.read_csv(r"C:\Users\asus\Downloads\test.csv", sep=';') #reading test dataset
```


```python
bankmd=pd.concat([train,test],join='inner', ignore_index=True) #combining train and test dataset
```


```python
bankmd.head() #getting peak of first five observations of the dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
bankmd.tail() #getting peak of last five observations of the dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49727</th>
      <td>33</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>-333</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>30</td>
      <td>jul</td>
      <td>329</td>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>49728</th>
      <td>57</td>
      <td>self-employed</td>
      <td>married</td>
      <td>tertiary</td>
      <td>yes</td>
      <td>-3313</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>9</td>
      <td>may</td>
      <td>153</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>49729</th>
      <td>57</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>295</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>aug</td>
      <td>151</td>
      <td>11</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>49730</th>
      <td>28</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1137</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>6</td>
      <td>feb</td>
      <td>129</td>
      <td>4</td>
      <td>211</td>
      <td>3</td>
      <td>other</td>
      <td>no</td>
    </tr>
    <tr>
      <th>49731</th>
      <td>44</td>
      <td>entrepreneur</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1136</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>3</td>
      <td>apr</td>
      <td>345</td>
      <td>2</td>
      <td>249</td>
      <td>7</td>
      <td>other</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
bankmd.shape #checking shape of the dataset
```




    (49732, 17)



The current dataset, after combining, has 49732 observations and 17 variables.


```python
bankmd.info() #checking general information about the dataset
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 49732 entries, 0 to 49731
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   age        49732 non-null  int64 
     1   job        49732 non-null  object
     2   marital    49732 non-null  object
     3   education  49732 non-null  object
     4   default    49732 non-null  object
     5   balance    49732 non-null  int64 
     6   housing    49732 non-null  object
     7   loan       49732 non-null  object
     8   contact    49732 non-null  object
     9   day        49732 non-null  int64 
     10  month      49732 non-null  object
     11  duration   49732 non-null  int64 
     12  campaign   49732 non-null  int64 
     13  pdays      49732 non-null  int64 
     14  previous   49732 non-null  int64 
     15  poutcome   49732 non-null  object
     16  y          49732 non-null  object
    dtypes: int64(7), object(10)
    memory usage: 6.5+ MB
    

The dataset has no null values. Out of 17 variables 10 has been read as object and 7 has been read as integers by Python. But here, the 'day' variable has read as integer which is incorrect.


```python
bankmd['day']=pd.to_datetime(bankmd['day']) #Hence changing data type of day, from integer to datetime
```


```python
bankmd.info() #rechecking general information after changing data type of day variable
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 49732 entries, 0 to 49731
    Data columns (total 17 columns):
     #   Column     Non-Null Count  Dtype         
    ---  ------     --------------  -----         
     0   age        49732 non-null  int64         
     1   job        49732 non-null  object        
     2   marital    49732 non-null  object        
     3   education  49732 non-null  object        
     4   default    49732 non-null  object        
     5   balance    49732 non-null  int64         
     6   housing    49732 non-null  object        
     7   loan       49732 non-null  object        
     8   contact    49732 non-null  object        
     9   day        49732 non-null  datetime64[ns]
     10  month      49732 non-null  object        
     11  duration   49732 non-null  int64         
     12  campaign   49732 non-null  int64         
     13  pdays      49732 non-null  int64         
     14  previous   49732 non-null  int64         
     15  poutcome   49732 non-null  object        
     16  y          49732 non-null  object        
    dtypes: datetime64[ns](1), int64(6), object(10)
    memory usage: 6.5+ MB
    

The 'day' variable has been treated. Now the dataset can be proceeded further.

### Verifying Duplicates


```python
bankmd.duplicated().sum() #checking if the dataset has any duplicate observations
```




    4521



The dataset has 4521 duplicate observations.


```python
bankmd.drop_duplicates(inplace=True) #dropping duplicates from the dataset
```

The duplicates has been dropped. 


```python
bankmd.shape #checking the shape of the dataset
```




    (45211, 17)



Now, the current dataset contains 45211 observations.

The dataset contains data of current and previous campaign. The variables 'pdays', 'previous' and 'poutcome' are the variables of the previous campaign.

The current campaign can be indentifid when the observations would contain :
<ol><li>pdays    - '-1'
    <li>previous - '0'
    <li>poutcome - 'unknown'</ol>
Now, classifying the dataset into current and previous dataset for further processing.


#### Creating Dataset of Current & Previous Campaign


```python
c_campaign=bankmd.copy(deep=True) #creating copy for current campaign
p_campaign=bankmd.copy(deep=True) #creating copy for previous campaign
```

##### Current Campaign


```python
c_campaign=c_campaign[c_campaign['pdays']==-1] #dataset with values only pdays equal to -1
c_campaign=c_campaign[c_campaign['previous']==0] #dataset with values only previous equals to 0
c_campaign=c_campaign[c_campaign['poutcome']=='unknown'] #dataset with only poutcome equals to unknown
```


```python
c_campaign.drop(columns=['pdays','previous','poutcome'],inplace=True) #dropping unnecessary columns
```


```python
c_campaign.shape #checking shape of the current campaign dataset
```




    (36954, 14)



The current campaign dataset contains 36954.

###### Checking The 'unknown' values in the Dataset


```python
c_campaign[c_campaign['job']=='unknown'] #checking presence of unknown values in variable job
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>216</th>
      <td>47</td>
      <td>unknown</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>28</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>338</td>
      <td>2</td>
      <td>no</td>
    </tr>
    <tr>
      <th>354</th>
      <td>59</td>
      <td>unknown</td>
      <td>divorced</td>
      <td>unknown</td>
      <td>no</td>
      <td>27</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>347</td>
      <td>3</td>
      <td>no</td>
    </tr>
    <tr>
      <th>876</th>
      <td>37</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>414</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000007</td>
      <td>may</td>
      <td>131</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>29</td>
      <td>unknown</td>
      <td>single</td>
      <td>primary</td>
      <td>no</td>
      <td>50</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000007</td>
      <td>may</td>
      <td>50</td>
      <td>2</td>
      <td>no</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>43653</th>
      <td>77</td>
      <td>unknown</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>397</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>1970-01-01 00:00:00.000000007</td>
      <td>may</td>
      <td>300</td>
      <td>3</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>44017</th>
      <td>57</td>
      <td>unknown</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>1884</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>1970-01-01 00:00:00.000000028</td>
      <td>jun</td>
      <td>133</td>
      <td>4</td>
      <td>no</td>
    </tr>
    <tr>
      <th>44681</th>
      <td>55</td>
      <td>unknown</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>159</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000004</td>
      <td>sep</td>
      <td>15</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>44714</th>
      <td>45</td>
      <td>unknown</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>406</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>1970-01-01 00:00:00.000000007</td>
      <td>sep</td>
      <td>314</td>
      <td>1</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>44742</th>
      <td>64</td>
      <td>unknown</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>2799</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>1970-01-01 00:00:00.000000009</td>
      <td>sep</td>
      <td>378</td>
      <td>4</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>255 rows × 14 columns</p>
</div>



The 'job' variable in the dataset contains unknown values.


```python
c_campaign[c_campaign['marital']=='unknown'] #checking presence of unknown values in variable marital
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



The 'marital' variable in the dataset does not contain unknown values.


```python
c_campaign[c_campaign['education']=='unknown'] #checking presence of unknown values in variable education
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>13</th>
      <td>58</td>
      <td>technician</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>71</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>71</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>16</th>
      <td>45</td>
      <td>admin.</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>13</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>98</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>42</th>
      <td>60</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>104</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>22</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>44804</th>
      <td>32</td>
      <td>student</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>0</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000015</td>
      <td>sep</td>
      <td>7</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>44816</th>
      <td>35</td>
      <td>management</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>2326</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>1970-01-01 00:00:00.000000016</td>
      <td>sep</td>
      <td>319</td>
      <td>1</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>44823</th>
      <td>20</td>
      <td>student</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>2785</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>1970-01-01 00:00:00.000000016</td>
      <td>sep</td>
      <td>327</td>
      <td>2</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>44871</th>
      <td>37</td>
      <td>blue-collar</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>217</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>1970-01-01 00:00:00.000000023</td>
      <td>sep</td>
      <td>272</td>
      <td>2</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>44900</th>
      <td>28</td>
      <td>admin.</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>174</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>1970-01-01 00:00:00.000000028</td>
      <td>sep</td>
      <td>184</td>
      <td>1</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>1534 rows × 14 columns</p>
</div>



The 'education' variable in the dataset contains unknown values.


```python
c_campaign[c_campaign['default']=='unknown'] #checking presence of unknown values in variable default
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



The 'marital' variable in the dataset does not contain unknown values.


```python
c_campaign[c_campaign['housing']=='unknown'] #checking presence of unknown values in variable housing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



The 'marital' variable in the dataset does not contain unknown values.


```python
c_campaign[c_campaign['loan']=='unknown'] #checking presence of unknown values in variable loan
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



The 'marital' variable in the dataset does not contain unknown values.


```python
c_campaign[c_campaign['contact']=='unknown'] #checking presence of unknown values in variable contact
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>92</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>198</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45027</th>
      <td>39</td>
      <td>services</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>471</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000015</td>
      <td>oct</td>
      <td>5</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45061</th>
      <td>30</td>
      <td>self-employed</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>1031</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000020</td>
      <td>oct</td>
      <td>7</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45062</th>
      <td>58</td>
      <td>retired</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>742</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000020</td>
      <td>oct</td>
      <td>5</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45122</th>
      <td>40</td>
      <td>entrepreneur</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>262</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000026</td>
      <td>oct</td>
      <td>17</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>45135</th>
      <td>53</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>1294</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000028</td>
      <td>oct</td>
      <td>71</td>
      <td>1</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>12950 rows × 14 columns</p>
</div>



The 'contact' variable in the dataset contains unknown values.

After checking the 'unknown' values in the dataset it can be seen that the dataset has three variables which contain the 'unknown' vaues, namely 'job','education' and 'contact'.<ol>
    <li>job - The subscription of the product can depend on the client's job, without the knowledge of his profession, marketing will be uneasy. Hence the uknown variables in this variable must be dropped.
        <li>educaion - The education of the client affects the subscription of the product, without knowing the education, marketing will be uneasy. Hence the unknown values in this variable must be dropped.
         <li>contact - The subsription of product does not rely on the means of contact but the duration of the contact, since the duration is given, dropping this variable will not help much.


```python
c_campaign['job'].replace('unknown',np.nan,inplace=True) #replacing the unknown values in job variable with nan values
c_campaign['education'].replace('unknown',np.nan,inplace=True)#replacing the unknown values in educaton variable with nan values
```


```python
c_campaign.isnull().sum() #checking the null values after replacing with the unknown values
```




    age             0
    job           255
    marital         0
    education    1534
    default         0
    balance         0
    housing         0
    loan            0
    contact         0
    day             0
    month           0
    duration        0
    campaign        0
    y               0
    dtype: int64




```python
c_campaign.dropna(axis=0,inplace=True) #dropping the null values from the dataset
```


```python
c_campaign.shape #checking shape after dropping null values
```




    (35281, 14)



The current campaign dataset is ready to proceed for further analysis.

##### Previous Campaign


```python
p_campaign=p_campaign[p_campaign['pdays']!=-1] #dataset with values only pdays not equals to -1
p_campaign=p_campaign[p_campaign['previous']!=0] #dataset with values only previous not equals to -1
p_campaign=p_campaign[p_campaign['poutcome']!='unknown'] #dataset with values only poutcome not equals to -1
```


```python
p_campaign.shape #shape of the previous campaign
```




    (8252, 17)



The previous campaign dataset has been filtered from the previous dataset.

### Current Campaign

#### DIstribution And Outliers


```python
c_campaign.head() #peek of the first five observations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>261</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>151</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>76</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>231</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>139</td>
      <td>1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>447</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>1970-01-01 00:00:00.000000005</td>
      <td>may</td>
      <td>217</td>
      <td>1</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



##### Varialbles Before Transformation


```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable
sns.histplot(data=c_campaign['age'],kde=True,color='orange')
plt.title("Age Disctribution") 
plt.xlabel('Age')
plt.ylabel('Total')
plt.show()
```


    
![png](output_61_0.png)
    



```python
c_campaign.age.skew() #checking skewness of the variable
```




    0.6161139918175859




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable
sns.histplot(data=c_campaign['balance'],kde=True,color='orange')
plt.title("Balance Disctribution")
plt.xlabel('Balance')
plt.ylabel('Total')
plt.show()
```


    
![png](output_63_0.png)
    



```python
c_campaign.balance.skew() #checking skewness of the variable
```




    8.451763998662019




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable
sns.histplot(data=c_campaign['duration'],kde=True,color='orange')
plt.title("Duration in Seconds Disctribution")
plt.xlabel('Duration')
plt.ylabel('Total')
plt.show()
```


    
![png](output_65_0.png)
    



```python
c_campaign.duration.skew() #checking skewness of the variable
```




    3.2610902166618616




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable
sns.histplot(data=c_campaign['campaign'],kde=True,color='orange')
plt.title("Campaign Disctribution")
plt.xlabel('Campaign')
plt.ylabel('Total')
plt.show()
```


    
![png](output_67_0.png)
    



```python
c_campaign.campaign.skew() #checking skewness of the variable
```




    4.58062128457041



Since, the variables of the dataset has negative values, cube root transformation would be a better approach to handle the skewness of the data.

##### Variables After Cube Root Transformation


```python
age_cbrt=np.cbrt(c_campaign['age']) #cube root transformation of the variable
age_cbrt.skew() #checking skewness of the variable
```




    0.24860229983957272




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable after cube root transformation
sns.histplot(data=age_cbrt,kde=True)
plt.title("Age Distribution")
plt.xlabel('Age in Cube Root')
plt.ylabel('Total')
plt.show()
```


    
![png](output_72_0.png)
    



```python
balance_cbrt=np.cbrt(c_campaign['balance']) #cube root transformation of the variable
balance_cbrt.skew() #checking skewness of the variable
```




    -0.05462258996416195




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable after cube root transformation
sns.histplot(data=balance_cbrt,kde=True)
plt.title("Balance Distribution")
plt.xlabel('Balance in Cube Root')
plt.ylabel('Total')
plt.show()
```


    
![png](output_74_0.png)
    



```python
duration_cbrt=np.cbrt(c_campaign['duration']) #cube root transformation of the variable
duration_cbrt.skew() #checking skewness of the variable
```




    0.6831847519620159




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable after cube root transformation
sns.histplot(data=duration_cbrt,kde=True)
plt.title("Duration Distribution")
plt.xlabel('Duration in Cube Root')
plt.ylabel('Total')
plt.show()
```


    
![png](output_76_0.png)
    



```python
campaign_cbrt=np.cbrt(c_campaign['campaign']) #cube root transformation of the variable
campaign_cbrt.skew() #checking skewness of the variable
```




    1.6043029780366664




```python
plt.figure(figsize=(7,5)) #plotting histogram to see the distribution and skewness of the variable after cube root transformation
sns.histplot(data=campaign_cbrt,kde=True)
plt.title("Campaign Distribution")
plt.xlabel('Camapaign in Cube Root')
plt.ylabel('Total')
plt.show()
```


    
![png](output_78_0.png)
    


###### Skewness Scale

|Range|Skewness|
|---|---|
|-0.5 to 0, 0 to 0.5|Fairly Symmetrical|
|-0.5 to -1, 0.5 to 1| Moderately Symmetrical|
|< -1, 1 > | Highly Skewed

###### Skewness Before & After Transformation

| Variables | Before Transformation | After Cube Root Transformation |
| --- | --- | --- |
| age | 0.61 | 0.24 |
| balance | 8.45 | -0.05 |
| duration | 3.26 | 0.68 |
| campaign | 4.58 | 1.60 |

75% of the numeric variables has been has turned to fairly symmetrical and moderately symmetrical, 25% of the variable is still highly skewed. In such cases, a non-parametric approach for further processing would be better. Here, the values after transformation is not taken for furhter processing.

#### Plotting graphs to see how the relation of customer of buying product varies


```python
plt.figure(figsize=(7,7))  #plotting pie chat to see the percentage of products suscribed in the current campaign
c_campaign['y'].value_counts().plot(kind='pie',autopct="%1.1f%%")
plt.title('Product Subscription')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_85_0.png)
    


fter plotting pie chart for checking the percentage of products suscribed in this campaign, it is found that % of products has been suscribed in this campaign of total  marketing calls.


```python
plt.figure(figsize=(7,5))  #plotting age and outcome into boxplot
sns.boxplot(data=c_campaign,x='y',y='age')
plt.title("Product Subscription Between Age & Outcome")
plt.xlabel('Subscription')
plt.ylabel('Age in Years')
plt.show()
```


    
![png](output_87_0.png)
    


Plotting Outcome and Age of the customer it can be seen that, 50% of customers who didn't suscribed the product are of age between 33 to 48. And 50% of customers who suscribed the product are of age between 32 to 50. 
From above results, we can say that we can not conlude if age between 30 to 50 is the actual reason of products not being suscribed because both subscibers and non-suscibers are following in this category. Hence another variable must be checked.


```python
plt.figure(figsize=(15,5)) #plotting bar graph in job to product subscribed in all job segment
sns.countplot(x=c_campaign['job'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Job')
plt.xlabel('Job')
plt.ylabel('Total')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_89_0.png)
    


From the results above, it is found that no specific interest has shown by any job segment to the product suscribed. However, the comparative percentage of 'retired', 'student' and 'unemployed' subscribing produducts are more.


```python
plt.figure(figsize=(7,5))    #plotting bar graph between marital status and outcome to see if product has subscribed
sns.countplot(x=c_campaign['marital'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Total')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_91_0.png)
    


After plotting bar graph between marital status and outcome to see if the product has been subscribed or not, it was seen that people who are married has shown more interest in subscribing the product, however, the percentage of single people subscribing the product are more.


```python
plt.figure(figsize=(7,5))   #plotting bar graph between education and outcome to see if it has affect product subscription
sns.countplot(x=c_campaign['education'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Education')
plt.xlabel('Eduaction')
plt.ylabel('Total')
plt.legend(title='Outcome')
plt.show()
```


    
![png](output_93_0.png)
    


After plotting bar graph in between education and the outcome, it is found that people who has completed only secondary eucation has been made more calls and the people completing secodary education has more suscribers. However, the comparatively, the people who has completed tertiary education has shown more interest towrads subscribing product.


```python
plt.figure(figsize=(7,5))   #plotting bar graph to between credit and outcome to see if products subscribed
sns.countplot(x=c_campaign['default'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Client Having Credit')
plt.xlabel('Credit')
plt.ylabel('Total')
plt.legend(title="Subscription")
plt.show()
```


    
![png](output_95_0.png)
    


Plotting the graph between the credit and outcome, it can be seen that, the credit does not heavily affectng the subscribing of the product. Even after no credit, people are subscribing at good comparative scale.


```python
plt.figure(figsize=(7,5))    #plotting boxplot between Outcome and Balance to see if product subscribed
sns.boxplot(data=c_campaign,y='balance',x='y')
plt.title("Product Subscription Along Balance of Customer Bank Account")
plt.xlabel('Subscription')
plt.ylabel('Balance')
plt.show()
```


    
![png](output_97_0.png)
    


After plotting boxplot between outcome and balance, it is found that the customers who are having balance less than or close to zero are less likely to subscibe product. The customers with balance more than zero has shown more interest in subscribing product.


```python
plt.figure(figsize=(7,5))  #plotting bar graph between housing loan and outcome to see if product subscribed
sns.countplot(x=c_campaign['housing'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Housing Loan')
plt.xlabel('Housing Loan')
plt.ylabel('Total')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_99_0.png)
    


Plotting a bar graph between housing loan and the the outcome, it is seen that the people who are having the housing loan are not much interested in subscribing products while people with no housing loan has shown comparatively more interest in subscribing the product.


```python
plt.figure(figsize=(7,5))    #plotting bar graph between personal loan and outcome to see if product subscribed
sns.countplot(x=c_campaign['loan'],hue=c_campaign['y'])
plt.title('Product Subscribed Over Personal Loan')
plt.xlabel('Personal Loan')
plt.ylabel('Total')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_101_0.png)
    


By plotting bar graph between personal loan and the outcome, it is seen that personal loans are less likely to affect the subscription of the product. The percentage of people subcribing product even after having personal loan or not, are around the same.


```python
plt.figure(figsize=(7,5)) #plotting bar graph between contact and outcome to see if product subscribed or not
sns.countplot(x=c_campaign['contact'],hue=c_campaign['y'])
plt.title('Product Subscription On Contact')
plt.xlabel('Contact')
plt.ylabel('Total')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_103_0.png)
    


After plotting bar graph between contact and outcome, it is found that contacting by telephone or cellular brings similar results. The contact medium variable has not shown any difference in telephone and cellular.


```python
plt.figure(figsize=(7,5))   #plotting boxplot between Outcome and campaign to see if product subscribed
sns.boxplot(data=c_campaign,x='y',y='campaign')
plt.title("Product Subscription Between Outcome & Campaign")
plt.xlabel('Subscription')
plt.ylabel('Campaign')
plt.show()
```


    
![png](output_105_0.png)
    


After plotting boxplot between outcome and campaign, it is found that the products subscribed and the campaign are negatively correlated. Campaign aroud 2 has more number of people subscribing the product.


```python
plt.figure(figsize=(7,6)) #plotting scatter plot between campaign and duration to see if see if product subscibed
sns.scatterplot(data=c_campaign,x='campaign',y='duration',hue='y')
plt.title('Relation Between Campaign & Duration Along Subscription')
plt.xlabel('Campaign')
plt.ylabel('Duration in Seconds')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_107_0.png)
    


After plotting scatter plot between campaign and duration of call made to the customer, along the outcome, it is found that the people with campaign less than 10 and the people with call duration more than 200 seconds are more likely to subscribe products.


```python
plt.figure(figsize=(7,6)) #plotting scatter plot between age and call duration to see if products subscribed or not
sns.scatterplot(data=c_campaign,x='age',y='duration',hue='y')
plt.title('Relation Between Duration & Age Along Subscription')
plt.xlabel('Age')
plt.ylabel('Duration in Seconds')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_109_0.png)
    


After plotting scatter plot between age and call duration in seconds, it is found that all age group has shown interest in subscribing the product when the call duration is more than 200 seconds.


```python
plt.figure(figsize=(7,6)) #plotting scatter plot between age and campaign to see if product subscribed
sns.scatterplot(data=c_campaign,x='age',y='campaign',hue='y')
plt.title('Relation Between Campaign & Age Along Subscription')
plt.xlabel('Age')
plt.ylabel('Campaign')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_111_0.png)
    


After plotting scatter plot between age and campaign, it is found that most age cateogirs has shown interest in subscribing the product with the campaign less than 10.


```python
plt.figure(figsize=(18,8))  #plotting boxplot between jobs and call duration to see if product subscribed
sns.boxplot(data=c_campaign,x='job',y='balance',hue='y')
plt.title("Product Subscription Between Job & Call Duration")
plt.xlabel('Job')
plt.ylabel('Balance')
plt.legend(title='Subscription')
plt.show()
```


    
![png](output_113_0.png)
    


After plotting boxplot between balance and the jobs, it is found that all job segments have subscribed products when the balance of the client's bank account is more than 0.


```python
plt.figure(figsize=(18,7)) #plottng bar graph between month and outcome
sns.countplot(x=c_campaign['month'],hue=c_campaign['y'])
plt.title('Product Subscription in Month')
plt.xlabel('Month')
plt.ylabel('Total')
plt.legend(title="Subscription")
plt.show()
```


    
![png](output_115_0.png)
    


After ploting bar graph between month and outcome, it can be seen that most of the products has been subscribed in the month of May. Since the dataset has no information about the year, no proper time trend can been shown in the product outcome across months.

#### Outcome Of The Previous Campaign


```python
plt.figure(figsize=(7,7))  #plotting pie chat to see the percentage of products suscribed in this campaign with previous dataset
p_campaign['poutcome'].value_counts().plot(kind='pie',autopct="%1.1f%%")
plt.title('Product Subscription in Previous Campaign')
plt.legend(title='Outcome')
plt.show()
```


    
![png](output_118_0.png)
    


In the previous campaign, the success rate of the outcome, i.e., product subscription is more than the current campaign.

### Findings

After analysing the dataset, it is found that there are few variables on which the subscription or outcome of the product depends on.The variables are as follows:<ul>
    <li>balance  : The balance of the client affects the subscription of the product, balance more than 0 increases the chances of subscription.
    <li>campaign : The number of times a client has been contacted affects the subscription of the product, campaign less than 10 increases the chances of the subscription.
    <li>duration : The duration of the call to a client affects the subscription of the product, duration more than 200 seocnds increases the chances of product subscrption. </ul>
To ensure the subscription of the product, the meausres in the following variables must be taken care of,
<ol>
    <li>Balance  : The balance amount of the client bank balance must be greater tha zero.
    <li>Campaign : The campaign of contact with the clients should keep minimum(not less than 1).
    <li>Duration : The duration of contact with the clients should be focused and it should be more than 250 seocnds.</ol>
Apart from these key variables, the marketing call should be prioritized with the following,
<ul><li>Retired and students client
    <li>Clients with No personal and home loan
    <li>Clients with tertiary education

These are the factors that would decide if the client will subscribe a term deposit or not.

#### When key factors are taken into consideration

If the three key factors are followed during marketing, it would result into the following subscription of the product:


```python
campaign=c_campaign.copy(deep=True)  #copying the c_campaign dataset 
```


```python
campaign=campaign[campaign['balance']>=0]  #picking all the observations having balance more than zero
```


```python
campaign=campaign[campaign['campaign']<=10] #picking all the observations having campaign less than ten
```


```python
campaign=campaign[campaign['duration']>=200] #picking all the observations having duration more than 200 seconds
```

The new dataset bankmd_new has is ready with the variables and the factors that would give more subscribtion than the previous bankmd_drop dataset.


```python
campaign.shape #checking the shape of the new dataset
```




    (14099, 14)



The new dataset has 3013 observations and 17 variables.


```python
plt.figure(figsize=(7,7))  #plotting pie chat to see the percentage of products suscribed in this campaign with new dataset
campaign['y'].value_counts().plot(kind='pie',autopct="%1.1f%%")
plt.title('Product Subscription When Founded Key Factors Taken Care Of')
plt.legend(title='Subscipton')
plt.show()
```


    
![png](output_132_0.png)
    


After taking care of the measures of the variable, it can be seen that there has been a growth of 9.1%, from 9.1% to 18.2%, in the subscription of the products.
