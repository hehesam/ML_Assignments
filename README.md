# **Supervised Learning with KRLS and Gaussian Kernel**

## **Project Overview**
This project demonstrates the implementation of a supervised learning algorithm using Kernel Regularized Least Squares (KRLS) with a Gaussian kernel. The objective is to perform multiclass classification using the provided dataset while optimizing hyperparameters via Q-fold cross-validation.

## **Project Contributors**
- **Bahaar Khalilian**
- **Hesam Mohebi**

## **Dataset Information**
- **X_train Shape**: (5200, 11), Data type: float64
- **Y_train Shape**: (5200, 1), Data type: int64
- **Classes**: 7 unique integer labels (3, 4, 5, 6, 7, 8, 9)

### **Observations:**
- The dataset is imbalanced, with classes 6 and 5 being the most prevalent and classes 3, 4, and 9 having fewer samples.

## **Implementation Details**
### **Key Steps:**
1. **Pre-processing**:
    - Standardized features using z-score normalization.
    - One-hot encoded labels for multiclass classification.
2. **KRLS Implementation**:
    - Trained individual KRLS models for each class in a one-vs-all (OvA) manner.
    - Used Gaussian kernel for KRLS.
3. **Cross-Validation**:
    - Performed 5-fold cross-validation to optimize the regularization parameter (λ) and Gaussian kernel parameter.
4. **Model Comparison**:
    - Compared KRLS with other models (Random Forest, SVM, Neural Network) using validation MSE.

## **Optimal Hyperparameters**
- **Regularization Parameter (λ)**: 0.0001
- **Gaussian Kernel Parameter**: 0.5

## **Results**
### **KRLS Performance:**
- **Best Cross-Validation Error**: 45.58%
- **Training Error**: 0.0%
- **Final Validation MSE**: 44.13%

### **Comparison with Other Models:**
| Model            | Validation MSE |
|------------------|----------------|
| KRLS             | 44.13%         |
| Random Forest    | 46.16%         |
| SVM              | 120%           |
| Neural Network   | 58.16%         |

## **Repository Structure**
```
|-- part2.ipynb             # Contains further KRLS implementation details
|-- Other_models.ipynb      # Implementation and results of Random Forest, SVM, and Neural Network
|-- Project Report.pdf      # Detailed project report
|-- trainingx.csv           # Training dataset features
|-- trainingy.csv           # Training dataset labels
```

## **Prerequisites**
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`

## **How to Run**
1. Clone the repository.
2. Install required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in the following order:
   - `part2.ipynb` for KRLS implementation.
   - `Other_models.ipynb` for Random Forest, SVM, and Neural Network.

## **Lessons Learned**
- The choice of kernel and regularization parameters significantly impacts KRLS performance.
- Handling class imbalance is crucial in multiclass classification tasks.
- Cross-validation is essential for robust hyperparameter tuning.

## **Future Work**
- Explore other kernels for KRLS (e.g., polynomial, Laplacian).
- Implement ensemble methods to combine KRLS with other models.
- Investigate advanced techniques for handling class imbalance (e.g., SMOTE).

## **References**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- Project Report: `Project Report.pdf`

---
For further questions or contributions, feel free to contact the contributors.

