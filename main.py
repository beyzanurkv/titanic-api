#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# In[2]:


app = FastAPI()


# In[3]:


# 4 dosyayı yükle (API başlarken bir kez çalışır)
imp             = joblib.load("model/imputer.pkl")
scaler          = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
model           = load_model("model/titanic_model.h5")


# In[4]:


class Yolcu(BaseModel):
    Pclass: int
    Sex: str
    Age: float = None    # None olabilir — eksik veri gelebilir
    SibSp: int
    Parch: int
    Fare: float = None
    Embarked: str = "S"
    Title: str = "Mr"


# In[5]:


def veri_hazirla(yolcu: Yolcu):
    d=yolcu.dict()
    df=pd.DataFrame([d])

    df=pd.get_dummies(df, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col]= 0

    df=df[feature_columns] 

    impute_cols=["Age","Fare"]
    mevcut=[c for c in impute_cols if c in df.columns]
    df[mevcut]=imp.transform(df[mevcut])

    df_scaled = scaler.transform(df)
    return df_scaled


# In[6]:


@app.get("/")
def anasayfa():
    return{"mesaj": "Titanic hayatta kalma API'si çalışıyor."}

@app.post("/predict")
def tahmin_et(yolcu: Yolcu):
    df_scaled=veri_hazirla(yolcu)
    olasilik=float(model.predict(df_scaled)[0][0])
    sonuc=int(olasilik>0.5)
    return {
        "hayatta_kalir_mi":sonuc,
        "olasilik":round(olasilik,3),
        "yorum":"Hayatta kalır" if sonuc ==1 else "Hayatta kalamaz"
    }


# In[ ]:




