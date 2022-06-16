# Enable HalvingSearch
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from warnings import simplefilter

def reclipper_scorer(estimator, X_train, y_train):
    # Computes R2 score, but clips y_train again, as it was done for training
    reclipped_y =  estimator['trf_reg'].transformer.transform(y_train)
    return r2_score(reclipped_y, estimator.predict(X_train))

class HyperparameterSearch:
    def generate_clip_dicts(self, first, last, step):
        # Generates a list of dictionaries with keys 'a_max' and 'a_min'
        # to use in grid search as kw_args of np.clip function
        # Will use range(first, last+1, step)
        list_of_dicts = []
        for a in range(first, last+1, step):
            a_dict = {}
            a_dict['a_min'] = 0
            a_dict['a_max'] = a
            list_of_dicts.append(a_dict)
        return list_of_dicts
    
    def run_HR_GS(self, base_model, X_train, y_train, param_distributions, 
                  print_best=True, ignore_warnings=False, 
                  scorer=reclipper_scorer):
        search = HalvingRandomSearchCV(base_model, param_distributions,
                                       min_resources=500, scoring=scorer, 
                                       random_state=42, verbose=1)

        if (ignore_warnings): simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X_train, y_train);
        if (ignore_warnings):simplefilter("default", category=ConvergenceWarning)

        if(print_best): print("Best params: ", search.best_params_)
        return search.best_estimator_
    
    def run_GS(self, base_model, X_train, y_train, param_distributions, 
               print_best=True, ignore_warnings=False, scorer=reclipper_scorer):
        search = GridSearchCV(base_model, param_distributions,
                              scoring=scorer,verbose=1)

        if (ignore_warnings): simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X_train, y_train);
        if (ignore_warnings):simplefilter("default", category=ConvergenceWarning)

        if(print_best): print("Best params: ", search.best_params_)
        return search.best_estimator_