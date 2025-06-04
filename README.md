
# K-Nearest Neighbors (KNN) Classification with Iris Dataset

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm on the famous **Iris flower dataset** using Python and Scikit-learn.

## Objective

- Understand and implement KNN for multi-class classification.
- Normalize features to improve distance-based learning.
- Experiment with different values of `K`.
- Evaluate model accuracy and interpret confusion matrix.
- Visualize decision boundaries using PCA.

---

##  Dataset

We use the built-in **Iris dataset** from Scikit-learn. It contains 150 samples of iris flowers from three species (*setosa*, *versicolor*, and *virginica*), with the following features:

| Feature            | Description                   |
|--------------------|-------------------------------|
| sepal length (cm)  | Sepal length in cm            |
| sepal width (cm)   | Sepal width in cm             |
| petal length (cm)  | Petal length in cm            |
| petal width (cm)   | Petal width in cm             |

The **target** variable is the species of the iris flower.

---

##  Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

---

##  Steps Implemented

1. Load and explore the Iris dataset.
2. Normalize features using `StandardScaler`.
3. Split data into training and testing sets.
4. Train `KNeighborsClassifier` with multiple values of `k` (1 to 10).
5. Plot accuracy vs. K to find the optimal `k`.
6. Evaluate the best model with:
   - Accuracy
   - Confusion Matrix
   - Classification Report
7. Visualize decision boundaries using PCA (2D).

---

##  Results

- Achieved **~100% accuracy** for `k = 3` on the test set.
- Visualized clear decision boundaries using PCA.
- Identified optimal K value based on validation accuracy.

---

## Visualizations

- **Accuracy vs. K** plot
- **Confusion Matrix** heatmap
- **Decision Boundaries** using PCA:

![KNN Decision Boundary](https://upload.wikimedia.org/wikipedia/commons/e/e7/Iris_dataset_scatterplot.svg)

---

##  How to Run

> You can run this in **Google Colab** or locally in Jupyter Notebook.

1. Clone the repository or copy the notebook file.
2. Run the notebook.
3. No dataset download required — Iris is loaded from `sklearn.datasets`.

---

##  Learnings

- KNN is a simple yet powerful non-parametric model.
- Normalizing features is crucial for distance-based models.
- Visualization helps understand model behavior in lower dimensions.

---

##  Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
