from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import xgboost as xgb


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
class ModifiedXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, params, classes):
        self.params=params
        self.classes=classes
        

    def fit(self, X, y=None):
        self.class_dict={cls:index for index,cls in enumerate(self.classes)}
        
        self.boost = xgb.XGBRegressor(objective='multi:softmax', num_class=6)
        
        params=self.params
        self.boost.set_params(**params)
        
        y=y.applymap(lambda x: self.class_dict[x])
        
        self.boost.fit(X,y)
        
        return self

    def predict(self, X):
        y=self.boost.predict(X)
        y=[self.classes[int(x)] for x in y]
        return y
