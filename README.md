# Telco Customer Churn Prediction API

Bu proje, Telco Customer Churn veri seti üzerinde müşteri kaybı tahmini yapan bir makine öğrenmesi modeli ve bu modeli kullanan basit bir FastAPI servisinden oluşur.

## Proje Amacı

Amaç, bir telekom müşterisinin hizmeti bırakıp bırakmayacağını (`Churn`) tahmin etmektir. Problem ikili sınıflandırma problemidir:

- `0`: Churn yok / müşteri ayrılmaz
- `1`: Churn var / müşteri ayrılır

## Kullanılan Yaklaşım

Model tarafında Random Forest kullanılmıştır. Preprocessing (ön işleme) adımları modelle birlikte tek bir `Pipeline` içine alınmıştır.

Pipeline içinde:

- Eksik sayısal değerler median ile doldurulur.
- Sayısal kolonlar olduğu gibi bırakılır (passthrough).
- Kategorik kolonlar OneHotEncoder ile dönüştürülür.
- Random Forest modeli eğitilir.

Bu sayede API tarafında tekrar manuel encoding veya scaling yapılmasına gerek kalmaz.

## Klasör Yapısı

```text
telco_churn_api_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── random_forest_churn_pipeline.pkl  
│   └── metrics.json                      
├── notebooks/
│   └── model.ipynb
├── example_request.json
├── index.html
├── requirements.txt
├── train_model.py
└── README.md
```

## Kurulum

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Paketleri kurun:

```bash
pip install -r requirements.txt
```

## Model Eğitimi

```bash
python train_model.py
```

Bu komut şu dosyaları üretir:

```text
models/random_forest_churn_pipeline.pkl
models/metrics.json
```

## API'yi Çalıştırma

```bash
uvicorn app.main:app --reload
```

API çalıştıktan sonra tarayıcıdan Swagger arayüzüne gidin:

```text
http://127.0.0.1:8000/docs
```

## Predict Endpoint

Endpoint:

```text
POST /predict
```

Örnek input:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

Örnek output:

```json
{
  "prediction": 1,
  "churn": "Yes",
  "churn_probability": 0.73,
  "model": "Random Forest"
}
```

## Curl ile Test

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d @example_request.json
```

## Değerlendirme Metrikleri

Model eğitildikten sonra `models/metrics.json` dosyasında şu metrikler yer alır:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Best Params

## Notlar

- API'ye `customerID` ve `Churn` gönderilmez.
- API input kolonları eğitimde kullanılan kolonlarla aynı tutulmuştur.
- Veri ön işleme aşamaları ve Random Forest modeli, kullanım kolaylığı için tek bir Pipeline olarak kaydedilmiştir.
My contribution
