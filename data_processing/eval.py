import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

class Evaluation:
    def rmse(self, y_true, y_pred):
        mse = mean_squared_error(y_pred, y_true)
        return np.sqrt(mse)

    def show_result(self, y_true, y_pred):
        print(f"R2={r2_score(y_true, y_pred):.3f},"+
              f"RMSE={-self.rmse(y_true, y_pred):.3f}")

    def show_result_cv(self, y_true, X_train, model):
        r2_res = cross_val_score(model, X_train, y_true, 
                                 cv=5, scoring='r2').mean()
        rmse_res = cross_val_score(model, X_train, y_true, cv=5, 
                                scoring='neg_root_mean_squared_error').mean()
        print(f"(CV) R2={r2_res:.3f},"+
              f"RMSE={rmse_res:.3f}")
    
    def print_training_results(self, model, X_train, y_train):
        self.show_result(y_train, model.predict(X_train))
        self.show_result_cv(y_train, X_train, model)
    
    def print_test_results(self, model, X_test, y_test):
        self.show_result(y_test, model.predict(X_test))