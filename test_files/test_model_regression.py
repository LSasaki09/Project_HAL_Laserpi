

import numpy as np
import time
import cv2
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import RBFInterpolator

#################### TEST MODEL POYLNOMIAL MULTIVARIATE POLYNOMIAL REGRESSION ####################
def train_polynomial_models(mm_xy, bit_xy, degree=3):
    """Entraîne des modèles pour prédire x_bits et y_bits à partir de (x_mm, y_mm)."""
    model_x = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_y = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_x.fit(mm_xy, bit_xy[:, 0])  # Pour x_bits
    model_y.fit(mm_xy, bit_xy[:, 1])  # Pour y_bits
    joblib.dump(model_x, 'model_x.pkl')
    joblib.dump(model_y, 'model_y.pkl') 
    return model_x, model_y

def project_mm_to_bits_pm(x_mm, y_mm, model_x, model_y):
    """Projette (x_mm, y_mm) en (x_bits, y_bits) avec les modèles entraînés."""
    X = np.array([[x_mm, y_mm]])
    bit_x = model_x.predict(X)[0]
    bit_y = model_y.predict(X)[0]
    return int(bit_x), int(bit_y)



######################RBF METHOD#######################
# Fonction d'entraînement pour RBFInterpolator
def train_rbf_models(mm_xy, bit_xy, kernel='thin_plate_spline', smoothing=0.0):
    """
    Entraîne des modèles RBF pour prédire x_bits et y_bits à partir de (x_mm, y_mm).
    
    Paramètres :
    - mm_xy : Tableau numpy de forme (N, 2) avec [x_mm, y_mm].
    - bit_xy : Tableau numpy de forme (N, 2) avec [x_bits, y_bits].
    - kernel : Type de noyau RBF ('thin_plate_spline', 'gaussian', etc.).
    - smoothing : Paramètre de lissage (0.0 pour interpolation exacte).
    
    Retourne :
    - model_x, model_y : Modèles RBF pour x_bits et y_bits.
    """
    model_x = RBFInterpolator(mm_xy, bit_xy[:, 0], kernel=kernel, smoothing=smoothing)
    model_y = RBFInterpolator(mm_xy, bit_xy[:, 1], kernel=kernel, smoothing=smoothing)
    joblib.dump(model_x, 'model_x_rbf.pkl')
    joblib.dump(model_y, 'model_y_rbf.pkl')
    print("RBF models saved to model_x_rbf.pkl and model_y_rbf.pkl")
    return model_x, model_y

# Nouvelle fonction de projection avec RBFInterpolator
def project_mm_to_bits_rbf(x_mm, y_mm, model_x, model_y):
    """
    Projette (x_mm, y_mm) en (x_bits, y_bits) avec des modèles RBF entraînés.
    
    Paramètres :
    - x_mm, y_mm : Coordonnées en millimètres.
    - model_x, model_y : Modèles RBF pour x_bits et y_bits.
    
    Retourne :
    - (bit_x, bit_y) : Coordonnées en bits (entiers).
    """
    X = np.array([[x_mm, y_mm]])
    try:
        bit_x = model_x(X)[0]
        bit_y = model_y(X)[0]
        return int(bit_x), int(bit_y)
    except Exception as e:
        print(f"Error in RBF interpolation: {e}")
        return None, None

###################R########################

# Fonction d'entraînement pour Random Forest
def train_rf_models(mm_xy, bit_xy, n_estimators=100, max_depth=10, min_samples_leaf=5):
    """
    Entraîne un modèle Random Forest pour prédire x_bits et y_bits à partir de (x_mm, y_mm).
    
    Paramètres :
    - mm_xy : Tableau numpy de forme (N, 2) avec [x_mm, y_mm].
    - bit_xy : Tableau numpy de forme (N, 2) avec [x_bits, y_bits].
    - n_estimators : Nombre d'arbres dans la forêt.
    - max_depth : Profondeur maximale des arbres.
    - min_samples_leaf : Nombre minimum d'échantillons par feuille.
    
    Retourne :
    - model : Modèle Random Forest pour les deux sorties (x_bits, y_bits).
    """
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
    model.fit(mm_xy, bit_xy)
    joblib.dump(model, 'model_rf.pkl')
    print("Random Forest model saved to model_rf.pkl")
    return model

# Nouvelle fonction de projection avec Random Forest
def project_mm_to_bits_rf(x_mm, y_mm, model):
    """
    Projette (x_mm, y_mm) en (x_bits, y_bits) avec un modèle Random Forest.
    
    Paramètres :
    - x_mm, y_mm : Coordonnées en millimètres.
    - model : Modèle Random Forest entraîné.
    
    Retourne :
    - (bit_x, bit_y) : Coordonnées en bits (entiers).
    """
    X = np.array([[x_mm, y_mm]])
    try:
        bit_xy = model.predict(X)[0]
        return int(bit_xy[0]), int(bit_xy[1])
    except Exception as e:
        print(f"Error in Random Forest prediction: {e}")
        return None, None

######################################################


#x_bit, y_bit = project_mm_to_bits_pm(x_mm, y_mm, model_x, model_y)
#x_bit, y_bit = project_mm_to_bits_rbf(x_mm, y_mm, model_x, model_y)
#x_bit, y_bit = project_mm_to_bits_rf(x_mm, y_mm, model_rf) #random forest


# Train polynomial models
#model_x, model_y = train_polynomial_models(mm_xy, bit_xy, degree=3)

# Load models
#model_x = joblib.load('model_x.pkl')
#model_y = joblib.load('model_y.pkl')  

# Entraîner les modèles RBF
#model_x, model_y = train_rbf_models(mm_xy, bit_xy, kernel='thin_plate_spline', smoothing=0.0)
#model_x = joblib.load('model_x_rbf.pkl')
#model_y = joblib.load('model_y_rbf.pkl')  

# Entraîner le modèle Random Forest
#model_rf = train_rf_models(mm_xy, bit_xy, n_estimators=100, max_depth=10, min_samples_leaf=5)

