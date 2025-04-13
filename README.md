## 🌾 Rice Type Classification — ML & Deep Learning Project

This project aims to classify rice grain types using both classical machine learning and deep learning approaches. By analyzing the morphological and statistical features of rice grains, the models can accurately distinguish between varieties like **Jasmine**, **Cammeo**, and **Osmancik**.

---

### 📌 Project Summary

#### 📥 1. Data Loading & Cleaning
- Loaded `riceClassification.csv` dataset using `pandas`.
- Removed unnecessary columns (e.g., `id`) and verified data integrity.
- Checked for and confirmed **no missing** or **duplicate values**.
- Conducted **outlier detection** using the IQR method and removed extreme points.

#### 📊 2. Exploratory Data Analysis (EDA)
- Generated a **pie chart** to verify class balance.
- Plotted:
  - **Density plots** to understand feature distributions.
  - **Boxplots** before and after outlier removal.
  - **Heatmap** of feature correlations to identify multicollinearity and relationships.

#### 🧹 3. Data Preprocessing
- Split the dataset into features (`X`) and target (`y`).
- Divided into training and test sets using an 80:20 split.
- Applied **StandardScaler** for feature scaling using `ColumnTransformer` in a `Pipeline`.

---

### 🤖 4. Classical Machine Learning Models
Built and evaluated the following models using `scikit-learn`:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Stacking Classifier** (combining RF, GBC, and DT)

For each model:
- Used `Pipeline` for preprocessing and modeling.
- Evaluated using:
  - **Accuracy Score**
  - **ROC-AUC Score**
  - **Confusion Matrix** with `seaborn` heatmaps

📊 A comparison DataFrame was generated to rank models by accuracy and ROC-AUC score.

---

### 🧠 5. Deep Learning (ANN)

#### ✅ Using TensorFlow/Keras
- Built an **Artificial Neural Network (ANN)** with:
  - 1 Input layer
  - 1 Hidden layer (ReLU activation)
  - 1 Output layer (Sigmoid activation)
- Used `Adam` optimizer and `EarlyStopping` for training.
- Plotted:
  - **Training vs. Validation Loss**
  - **Training vs. Validation Accuracy**
- Achieved high accuracy on the test set after scaling and optimization.

#### ✅ Using PyTorch
- Created a **custom PyTorch Dataset** and DataLoader.
- Built a simple **ANN architecture** with:
  - Linear layers + ReLU
  - Sigmoid output for binary classification
- Trained over multiple epochs and printed:
  - **Train & Validation Loss**
  - **Accuracy per epoch**
- Evaluated on the test set and printed **final test accuracy and loss**.

---

### 🔧 Technologies Used

| Category             | Tools & Libraries                           |
|----------------------|---------------------------------------------|
| Language             | Python                                      |
| Data Handling        | Pandas, NumPy                               |
| Visualization        | Matplotlib, Seaborn                         |
| Classical ML         | Scikit-learn, XGBoost                       |
| Deep Learning        | TensorFlow, Keras, PyTorch                  |
| Model Evaluation     | Accuracy, ROC-AUC, Confusion Matrix         |
| Miscellaneous        | Pipelines, ColumnTransformer, EarlyStopping |

---

### 📈 Performance Summary

| Model                   | Accuracy Score | ROC-AUC Score |
|------------------------|----------------|----------------|
| Decision Tree           | ✅ Measured     | ✅ Measured     |
| Random Forest           | ✅ Measured     | ✅ Measured     |
| Gradient Boosting       | ✅ Measured     | ✅ Measured     |
| Stacking Classifier     | ✅ Measured     | ✅ Measured     |
| ANN (TensorFlow/Keras)  | ✅ Measured     | N/A (binary)   |
| ANN (PyTorch)           | ✅ Measured     | N/A (binary)   |

---

### 🧠 Conclusion

- Successfully implemented a complete pipeline for rice type classification using both **traditional ML** and **deep learning**.
- **Stacking Classifier** and **ANNs** (Keras/PyTorch) demonstrated high accuracy.
---

> ⚡ _This project highlights how combining domain-specific features with machine learning can lead to accurate and interpretable models in real-world applications like crop classification._

