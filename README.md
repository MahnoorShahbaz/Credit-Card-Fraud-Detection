## Credit-Card-Fraud-Detection

## Team Members
-Mahnoor Shahbaz (CMS ID: 524267)
-Areeba Sohail (CMS ID: 517497)

## Course Information
-Course:CS 470-Machine Learning
-Semester: Fall 2025
-Department: School of Electrical and Computer Engineering, SEECS
-Institution: National University of Sciences and Technology (NUST), Pakistan
-Instructor: Sir Sajjad Hussain

## Repository Structure
1) notebooks/: Contains the core implementations.
   -ML_Project_Deep_Learning.ipynb: The implementation of a deep learning model, using neural networks.
   -ML_Project_Logistic_Regression.ipynb: One implementation of a classical machine learning approach, using logistic regression.
   -ML_Project_XG_Boost.ipynb: Our second implementation of a classical machine learning approach, using XG Boost.
2)license: Standard project licensing. 
3)README.md: Project documentation, results summary, and execution instructions.

## Abstract
The objective is to detect credit card fraud. We will be using two classical machine learning models and a singular deep learning implementation to identify fraudulent credit card transactions in highly imbalanced datasets. Then, we will lead a comparative analysis on it to compare performance and identify the most effective approach using metrics like precision, recall, F1-score, and AUC-ROC. Our results demonstrate that Deep Neural Network achieved the best performance with 90.71% accuracy and 91.2% recall, effectively balancing the trade-off between detecting fraudulent transactions and minimizing false positives.Among classical ML approaches, XGBoost with Bayesian Optimization (BOA) outperformed Logistic Regression with 99.95% accuracy, 74% precision, and 76% recall. The deep learning model's superior recall (91.2%) indicates it successfully catches more fraudulent transactions while maintaining high precision (96.3%), making it the most suitable for production deployment in fraud detection systems.

## Problem Statement
This project addresses the critical challenge of credit card fraud, a global issue resulting in billions of dollars in annual losses. The primary technical hurdle involves navigating extreme class imbalance, where fraudulent transactions constitute less than 0.2% of the dataset, rendering traditional accuracy metrics misleading.

## Project Objectives
The primary objectives of this project are:
-Implement Classical ML Models: Develop and train at least two classical machine learning algorithms with proper hyperparameter tuning.
-Design Deep Learning Architecture: Create a neural network using TensorFlow/Keras with appropriate architecture for binary classification.
-Handle Class Imbalance: Apply techniques to address the severe imbalance between fraudulent and legitimate transactions.
-Comprehensive Evaluation: Compare model performance using multiple evaluation metrics appropriate for imbalanced classification.
-Real-world Application: Demonstrate practical applicability of the models for fraud detection systems.

## Significance
Effective fraud detection systems are crucial for:
-Financial Security: Protecting consumers and financial institutions from monetary losses.
-Customer Trust: Maintaining confidence in digital payment systems.
-Operational Efficiency: Reducing manual review costs through automated detection.
-Regulatory Compliance: Meeting industry standards for fraud prevention.

## How to Run
To run these notebooks, ensure archive(2).zip is uploaded to the root directory as demonstrated in the data loading cells.

## Dataset Used
-The dataset used in this project is the Credit Card Fraud Detection dataset available on Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud).
-Content: It contains transactions made by European cardholders in September 2013.
-Features: It includes 28 numerical features ($V1$ through $V28$) which are the result of a PCA transformation, alongside 'Time' and 'Amount'.
-Imbalance: The dataset is highly imbalanced, with only 492 frauds out of 284,807 transactions (0.172%).

## Models used
This project performs an ablation study (a process of testing different components) to compare different algorithmic approaches for fraud detection:
-Logistic Regression: Used as a to set a "minimum" performance score to beat.
-XGBoost: A powerful tree-based model specifically tuned for imbalanced tabular data.
-Deep Learning (Neural Network): A neural network (TensorFlow 2.x) designed to find hidden, non-linear fraud patterns.  

## Deep Learning Approach
Neural Network Architecture
Architecture Rationale:
-Hidden Layers: Progressive dimensionality reduction (128 → 64 → 32)
-Activation: ReLU for hidden layers to address vanishing gradients
-Batch Normalization: Stabilizes learning and accelerates training
-Dropout: Prevents overfitting (rates: 0.3, 0.3, 0.2)
-Output: Sigmoid for binary classification probability
<img width="920" height="683" alt="image" src="https://github.com/user-attachments/assets/9f593602-c2bb-48ae-8c22-ec5ca5725d3d" />

## Evaluation Metrics
Because the data is extremely imbalanced, standard "Accuracy" is not a reliable metric. Instead, we evaluate the models based on:
-Precision: How many of the predicted frauds were actually fraud?
-Recall (Sensitivity): What percentage of total actual frauds did the model catch?
-F1-Score: The harmonic mean of Precision and Recall.
-Precision-Recall AUC (PR-AUC): The primary metric used to evaluate model performance, as it is more robust than ROC-AUC for imbalanced data.

## Performance Metrics
-For XG Boost:
| Configuration | F1-Score | Precision | Recall | ROC-AUC | PR-AUC |
|---------------|----------|-----------|--------|---------|--------|
| Baseline 1 (Default) | 84% | 88% | 81% | 0.95 | 0.80 |
| Baseline 2 (+ SMOTE) | 79% | 73% | 86% | 0.99 | 0.88 |
| **Ablasi (+ BOA)** | **88%** | **95%** | **83%** | **0.97** | **0.89** |
| Proposed (+ SMOTE + BOA) | 75% | 74% | 76% | 0.98 | 0.78 |
Ablasi is highlighted since it achieved the best overall performance with 88% F1-score and 0.89 PR-AUC. The BOA hyperparameter optimization effectively handles class imbalance without requiring SMOTE. 
<img width="625" height="537" alt="image" src="https://github.com/user-attachments/assets/88f72569-0d63-4abf-bddf-7ec031b50005" />
<img width="625" height="537" alt="image" src="https://github.com/user-attachments/assets/acd20af5-2f80-4590-9189-71b90fe72bd6" />
<img width="546" height="536" alt="image" src="https://github.com/user-attachments/assets/81796b03-ea6e-456c-9a5e-3da876356b01" />
<img width="850" height="536" alt="image" src="https://github.com/user-attachments/assets/37e0e827-87d6-403a-b3f7-a00eee18a326" />
<img width="533" height="532" alt="image" src="https://github.com/user-attachments/assets/d3673e61-804e-4740-b8c9-8e04e6270e3c" />
<img width="1308" height="648" alt="image" src="https://github.com/user-attachments/assets/54bc0f7d-7bc2-4994-929b-db003732d0e7" />
<img width="967" height="570" alt="image" src="https://github.com/user-attachments/assets/7b8f7373-257d-4fed-aebb-d920d529445c" />



For Logistic Regression:
The curve shows a favorable precision-recall trade-off, with the model maintaining high precision until approximately 80% recall. Beyond this point, precision drops sharply, indicating that catching the remaining 20% of frauds would require accepting many more false positives. 
<img width="889" height="683" alt="image" src="https://github.com/user-attachments/assets/50e052da-0c59-46e8-998d-b2adb4aa862a" /> 
<img width="769" height="595" alt="image" src="https://github.com/user-attachments/assets/ec022e28-8430-4826-a4ef-367d34f8abfb" />


<img width="901" height="621" alt="image" src="https://github.com/user-attachments/assets/65462138-9e87-435c-b033-69f19b567ca5" />
<img width="1248" height="727" alt="image" src="https://github.com/user-attachments/assets/dc1b7232-5cfc-4a18-943d-22d735f0788f" />



| Metric | XG Boost | Logistic Regression | Deep Neural Network |
|--------|--------------|-----|---------------------|
| **Accuracy** | 99.95% | 99.93% | 90.71% |
| **Precision** | 74% | 66% | 96.3% |
| **Recall** | 76% | 79% | 91.2% |
| **F1-Score** | 0.75 | 0.72 | 0.935 |
| **AUC-ROC** | 97% | 96% | 98.3% |
| **AUC-PR** | 89% | 89% | 91.2% |

## Model Limitations
-Generalization: Performance on new fraud patterns uncertain
-Interpretability: Deep learning model lacks transparency
-Overfitting Risk: Despite regularization, may not generalize to future frauds
-Adversarial Robustness: Fraudsters may adapt to detection patterns

## Outcome
The final results of the study indicate:
-Sampling Impact: The use of SMOTE (Synthetic Minority Over-sampling Technique) significantly improved the Recall for all models, ensuring fewer fraudulent transactions went undetected.
-Top Performer: Deep Neural Network achieved the highest F1-Score of 0.93.
-Business Value: The final model successfully identifies fraud with a high degree of confidence, providing a balanced trade-off 
-Class Imbalance Handling is Critical: SMOTE and class weights dramatically improved minority class detection
-Metric Selection Matters: F1-score and AUC-PR more informative than accuracy for imbalanced problems
-Trade-offs Exist: Balance between catching frauds (recall) and avoiding false alarms (precision) requires business context
-Feature Engineering Limited: PCA transformation restricts domain knowledge integration
-Model Interpretability vs. Performance: Classical models offer transparency, deep learning offers potential performance gain

## Conclusion
This project successfully implemented and compared classical machine learning and deep learning approaches for credit card fraud detection on a highly imbalanced dataset. Through comprehensive experimentation and evaluation, we demonstrated that [best performing model] achieved the optimal balance between detecting fraudulent transactions and minimizing false positives, with [X]% recall and [Y]% precision.

## Final Recommendation:
For production deployment, we recommend the Deep Neural Network due to its superior fraud detection performance (91.2% recall, 96.3% precision, 93.5% F1-score, and 98.3% ROC-AUC). The model successfully identifies 91 out of 100 fraudulent transactions while maintaining minimal false positives, translating to substantial financial savings and enhanced customer protection. Despite requiring longer training time and reduced interpretability compared to classical models, the DNN's ability to capture complex non-linear fraud patterns justifies its deployment in high-stakes fraud detection systems.

However, a hybrid approach combining multiple models through ensemble methods or sequential screening could provide optimal results.

## Reproducibility
-To reproduce results:
1)Clone the repository:
git clone https://github.com/MahnoorShahbaz/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
2)Install dependencies:
pip install -r requirements.txt
Download dataset from Kaggle and place in data/ folder
Run notebooks in order (01 → 05)
Results will be saved in results/ directory.

## Team Contributions
Task,                        Mahnoor Shahbaz  ,  Areeba Sohail
Data Preprocessing,                   ✅     ,      ✅
EDA and Visualization,                ✅     ,      ✅
XG Boost,                             ✅     ,
Logistic Regression,                  ✅     ,
Deep Learning Model,                  ✅     ,      ✅
Hyperparameter Tuning,                ✅     ,      ✅
Evaluation and Analysis,              ✅     ,      ✅
Documentation,                        ✅     ,      

## Acknowledgments
We would like to thank:
-Dr. Sajjad Hussain for guidance throughout the project
-Kaggle and ULB Machine Learning Group for providing the dataset
-NUST SEECS for computational resources
-Open-source community for excellent libraries and tools

## Contact Information
For questions or collaboration:
-Mahnoor Shahbaz
Email:mahnoorshahbazameer@gmail.com
GitHub: @MahnoorShahbaz


-Areeba Sohail
Email: sohailareeba14@gmail.com

Project Repository: https://github.com/MahnoorShahbaz/Credit-Card-Fraud-Detection



