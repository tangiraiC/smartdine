
---

# **SmartDine: A Multimodal Restaurant Recommender System**

SmartDine is an end-to-end **multimodal recommender system** that predicts restaurant relevance using **textual reviews (MPNet)**, **images (CLIP)**, and **structured metadata**.
The system includes a complete **data pipeline**, **model training suite**, a **Django REST inference API**, and a **React demo interface**.

---

## **Project Features**

* Multimodal embeddings (text, image, dense features)
* Multiple recommender models: MF, NCF, unimodal towers, Late Fusion, Joint Embedding
* Precomputed item-level caches for efficient serving
* Real-time recommendation API (Django REST)
* React-based demonstration UI
* Offline diagnostics: PCA, t-SNE, modality ablation

---

## **System Architecture**

```
Frontend (React)
        |
        v
Django REST API
        |
        v
Late Fusion Recommender
(Text + Image + Dense)
        |
        v
Top-K Ranked Results
```

---

## **Installation**

### **1. Create environment**

```bash
conda create -n smartdine python=3.10
conda activate smartdine
pip install -r requirements.txt
```

### **2. Download NLTK resources**

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## **Data Processing Pipeline**

```bash
python recommender/scripts/build_text_embeddings.py
python recommender/scripts/build_image_embeddings.py
python recommender/scripts/build_dense_features.py
python recommender/scripts/build_item_feature_cache.py
```

---

## **Model Training**

```bash
# Baselines
python recommender/train/train_mf.py
python recommender/train/train_ncf.py

# Unimodal towers
python recommender/train/train_text_tower.py
python recommender/train/train_image_tower.py
python recommender/train/train_dense_tower.py

# Fusion models
python recommender/train/train_late_fusion.py   # (Deployed Model)
python recommender/train/train_joint_model.py
```

---

## **Running the API**

```bash
cd backend
python manage.py runserver
```

**Endpoint:**

```
POST /api/recommendations/
{
  "query_text": "sushi",
  "k": 5
}
```

---

## **Running the Frontend**

```bash
cd smartdine-frontend
npm install
npm start
```

---

## **Diagnostics**

```bash
python -m recommender.analysis.week5_diagnostics
```

Generates PCA, t-SNE, and score-popularity plots.

---

## **Authors**

* **Lincoln Chanakira**
* **Caleb Christian**

---

## **Citation**

```
@misc{smartdine2025,
  title={SmartDine: A Multimodal Restaurant Recommender System},
  author={Chanakira, Lincoln and Christian, Caleb},
  year={2025}
}
```

---

