# SmartDine: Technical Overview & Recommendation Logic

## 1. System Overview
SmartDine is a **Multimodal Recommendation System** that helps users discover restaurants using a hybrid approach. It combines:
1.  **Collaborative Filtering (Deep Learning)**: Learns user preferences and item characteristics from interaction history.
2.  **Content-Based Retrieval (Semantic Search)**: Uses Large Language Models (LLMs/Transformers) to understand the semantic meaning of user queries and match them with restaurant descriptions.

## 2. Architecture

### Backend (Django + PyTorch)
-   **API Framework**: Django Rest Framework (DRF) serves the recommendation endpoints.
-   **Inference Engine**: A custom `JointRecModel` (PyTorch) is loaded into memory to predict scores in real-time.
-   **Vector Search**: `Sentence-Transformers` (SBERT) encodes user queries on-the-fly to calculate similarity against pre-computed restaurant embeddings.

### Frontend (React)
-   **Interactive UI**: Allows users to input queries and tune the "mode" (Discovery vs. Search) via a slider.
-   **Visualization**: Displays "Insights" bars showing how much the Text Relevance vs. Model Quality contributed to the final score.

---

## 3. Recommendation Process (Step-by-Step)

When a user submits a query (e.g., *"spicy ramen"*) and a mode (weights), the following process occurs in `backend/recs/services.py`:

### Step 1: Feature Loading
The system lazily loads the necessary assets into memory:
-   **Trained Model**: `JointRecModel` weights (`joint_best.pt`).
-   **Item Cache**: Pre-computed features for all restaurants:
    -   `text_emb`: SBERT embeddings of descriptions/reviews (768-dim).
    -   `img_emb`: ResNet embeddings of food images (512-dim).
    -   `dense`: Numerical features (sales volume, ratings, etc.).
-   **Metadata**: Names, addresses, and image URLs.

### Step 2: Joint Model Scoring (The "Quality" Score)
The `JointRecModel` predicts how likely a user is to interact with a restaurant *regardless* of their current text query. 
-   **Input**: User ID (or dummy ID for cold-start), Item Embeddings (Text + Image + Dense).
-   **Mechanism**: These inputs are concatenated and passed through a Multi-Layer Perceptron (MLP) network.
-   **Output**: A scalar `base_score` (logit) representing general affinity/quality.

### Step 3: Semantic Similarity Scoring (The "Relevance" Score)
The system calculates how well the restaurant matches the specific text query.
-   **Query Encoding**: The user's text *"spicy ramen"* is encoded using `sentence-transformers/all-mpnet-base-v2` into a vector $q$.
-   **Cosine Similarity**: We calculate the dot product between the query vector $q$ and every restaurant's text embedding $d$.
    $$ \text{sim} = \frac{q \cdot d}{||q|| ||d||} $$
-   **Output**: A `sim` score between -1 and 1.

### Step 4: Weighted Fusion
The final score is a linear combination of the two components, controlled by the user's *Mode* slider:

$$ \text{FinalScore} = (W_{joint} \times \text{BaseScore}) + (W_{text} \times \text{SimilarityScore}) $$

-   **Discovery Mode**: High $W_{joint}$, Low $W_{text}$. Prioritizes high-quality items even if they don't perfectly match the text.
-   **Search Mode**: Low $W_{joint}$, High $W_{text}$. Prioritizes exact semantic matches.

### Step 5: Ranking & Serving
1.  All restaurants are sorted by `FinalScore`.
2.  The top $k$ results are returned to the frontend.
3.  The frontend displays the breakdown of `BaseScore` and `SimilarityScore` as "Quality" and "Relevance" bars.

---

## 4. Key Technologies / Files

-   **`recommender/models/joint.py`**: Defines the Neural Network architecture.
-   **`recommender/serving/query_encoder.py`**: Handles SBERT loading and text encoding.
-   **`backend/recs/services.py`**: The orchestration layer that loads models, computes scores, and ranks items.
-   **`smartdine-frontend/src/components/QueryForm.jsx`**: The UI control that converts user intent (Slider) into technical weights ($W_{text}$).
