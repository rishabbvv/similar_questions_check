# Duplicate Question Pair Detection

This project trains models to detect whether two questions mean the same thing.

It supports:

- TF-IDF / bag-of-words features with Logistic Regression
- TF-IDF / bag-of-words features with Random Forest
- Optional Word2Vec averaged sentence vectors
- Siamese BiLSTM neural network

The expected CSV columns are:

```text
question1,question2,is_duplicate
```



## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not on PATH in Codex, this bundled Python worked during setup:

```powershell
& "C:\dependencies\python\python.exe" -m pip install -r requirements.txt
```

## Train Classic ML Baselines

Train Logistic Regression with TF-IDF bag-of-words:

```powershell
python src/train_baseline.py --data "data/question-pair.csv" --model logistic --features tfidf
```

Train Random Forest with TF-IDF features reduced with SVD:

```powershell
python src/train_baseline.py --data "data/question-pair.csv" --model random_forest --features tfidf
```

Train Logistic Regression using Word2Vec averaged sentence vectors:

```powershell
python src/train_baseline.py --data "data/question-pair.csv" --model logistic --features word2vec
```

For a fast smoke test, train on a sample:

```powershell
python src/train_baseline.py --data "data/question-pair.csv" --sample-size 20000 --model logistic --features tfidf
```

## Train Siamese LSTM

```powershell
python src/train_siamese_lstm.py --data "data/question-pair.csv" --epochs 5 --batch-size 128
```

For a fast smoke test:

```powershell
python src/train_siamese_lstm.py --data "data/question-pair.csv" --sample-size 20000 --epochs 1
```

## Predict With a Saved Baseline Model

```powershell
python src/predict.py --model-path models/logistic_tfidf.joblib --q1 "How do I learn Python?" --q2 "What is the best way to study Python?"
```

## Run The LSTM Flask Website

The website loads `models/siamese_lstm.pt` and predicts with the trained Siamese LSTM.

```powershell
python app.py
```

If `python` is not on PATH, use:

```powershell
& "C:\codex-primary-runtime\dependencies\python\python.exe" app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Outputs

Trained models are saved in `models/`.

Metrics JSON files are saved in `reports/`.
