{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "29397761-bfba-4c8c-96f7-1c9a3c123b5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1ec4e98d-42be-4987-ba16-400ebc97fb63",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = FastAPI()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f690a78e-2fab-40f0-a53c-e8b5c0fc291e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "# 4 dosyayı yükle (API başlarken bir kez çalışır)\n",
    "imp             = joblib.load(\"model/imputer.pkl\")\n",
    "scaler          = joblib.load(\"model/scaler.pkl\")\n",
    "feature_columns = joblib.load(\"model/feature_columns.pkl\")\n",
    "model           = load_model(\"model/titanic_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "933c8895-c0c6-4fe4-b685-4df8f075064d",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Yolcu(BaseModel):\n",
    "    Pclass: int\n",
    "    Sex: str\n",
    "    Age: float = None    # None olabilir — eksik veri gelebilir\n",
    "    SibSp: int\n",
    "    Parch: int\n",
    "    Fare: float = None\n",
    "    Embarked: str = \"S\"\n",
    "    Title: str = \"Mr\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e89fd80b-6af7-493b-9bdd-8ea280ae9e89",
   "metadata": {},
   "outputs": [],
   "source": [
    "def veri_hazirla(yolcu: Yolcu):\n",
    "    d=yolcu.dict()\n",
    "    df=pd.DataFrame([d])\n",
    "\n",
    "    df=pd.get_dummies(df, drop_first=True)\n",
    "\n",
    "    for col in feature_columns:\n",
    "        if col not in df.columns:\n",
    "            df[col]= 0\n",
    "\n",
    "    df=df[feature_columns] \n",
    "\n",
    "    impute_cols=[\"Age\",\"Fare\",\"Pclass\",\"SibSp\",\"Parch\"]\n",
    "    mevcut=[c for c in impute_cols if c in df.columns]\n",
    "    df[mevcut]=imp.transform(df[mevcut])\n",
    "\n",
    "    df_scaled = scaler.transform(df)\n",
    "    return df_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "04d20ab0-5474-4d6e-a20b-eaf2857a9e23",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.get(\"/\")\n",
    "def anasayfa():\n",
    "    return{\"mesaj\": \"Titanic hayatta kalma API'si çalışıyor.\"}\n",
    "\n",
    "@app.post(\"/predict\")\n",
    "def tahmin_et(yolcu: Yolcu):\n",
    "    df_scaled=veri_hazirla(yolcu)\n",
    "    olasilik=float(model.predict(df_scaled)[0][0])\n",
    "    sonuc=int(olasilik>0.5)\n",
    "    return {\n",
    "        \"hayatta_kalir_mi\":sonuc,\n",
    "        \"olasilik\":round(olasilik,3),\n",
    "        \"yorum\":\"Hayatta kalır\" if sonuc ==1 else \"Hayatta kalamaz\"\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "846b5dd9-e950-40d7-945c-572dfb46479c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
