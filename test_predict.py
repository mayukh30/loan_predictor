import pickle
import os

models_dir = os.path.join(os.getcwd(), 'models')
with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
    lr_model = pickle.load(f)

print(type(lr_model))
print(hasattr(lr_model, 'multi_class'))

if not hasattr(lr_model, 'multi_class'):
    lr_model.multi_class = 'ovr'

print(hasattr(lr_model, 'multi_class'))

import numpy as np
try:
    # mock predict_proba
    print("Classes:", lr_model.classes_)
    # pass 27 features
    mock_input = np.zeros((1, 27))
    lr_model.predict_proba(mock_input)
except Exception as e:
    import traceback
    traceback.print_exc()
