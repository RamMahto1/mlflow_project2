from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split


def data_load():
    data=load_diabetes()
    df=pd.DataFrame(data.data, columns=data.feature_names)
    df['target']=data.target
    X = df.drop("target",axis=1)
    y = df['target']
    
    return train_test_split(X,y, test_size=0.20, random_state=42)
    
    
    