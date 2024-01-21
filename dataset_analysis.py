from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
import numpy as np

c3_data = scipy.io.loadmat('C:/Users/burak/OneDrive/Masaüstü/ÖRÜNTÜ2024_Proje/C3/wifi_localization.mat')
feat_c3 = c3_data['feat']
lbl_c3 = c3_data['lbl']

r3_data = scipy.io.loadmat('C:/Users/burak/OneDrive/Masaüstü/ÖRÜNTÜ2024_Proje/R3/Gas_Turbine_Co_NoX_2015.mat')
feat_r3 = r3_data['feat']
lbl_r3 = r3_data['lbl1']

print("C3 Veri Seti:")
print("feat_c3 shape:", feat_c3.shape)
print("lbl_c3 shape:", lbl_c3.shape)

print("\nR3 Veri Seti:")
print("feat_r3 shape:", feat_r3.shape)
print("lbl_r3 shape:", lbl_r3.shape)

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# Random Forest Classifier 
rf_model = RandomForestClassifier()
rf_scores = {
    'accuracy': cross_val_score(rf_model, feat_c3, lbl_c3.ravel(), cv=kfold, scoring='accuracy'),
    'f1_score': cross_val_score(rf_model, feat_c3, lbl_c3.ravel(), cv=kfold, scoring='f1_weighted')
}

print("\nRandom Forest Classifier Scores:")
print("Accuracy:", np.mean(rf_scores['accuracy']))
print("F1 Score:", np.mean(rf_scores['f1_score']))

# Lasso Regression
lasso_model = Lasso()
lasso_scores = {
    'mae': cross_val_score(lasso_model, feat_r3, lbl_r3.ravel(), cv=kfold, scoring='neg_mean_absolute_error')
}

print("\nLasso Regression Scores:")
print("MAE:", np.mean(lasso_scores['mae']))


