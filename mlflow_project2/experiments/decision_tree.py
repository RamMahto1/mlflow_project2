from utils.data_loader import data_load
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def run():
    X_train, X_test, y_train,y_test= data_load()
    
    with mlflow.start_run(run_name="DecisionTreeRegressor"):
        mlflow.set_tag("model_name", "DecisionTreeRegressor")
        model=DecisionTreeRegressor()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        
        
        ## model evaluation
        mae=mean_absolute_error(y_test,prediction)
        mse=mean_squared_error(y_test,prediction)
        r2_square=r2_score(y_test,prediction)
        rmse=np.sqrt(mean_squared_error(y_test,prediction))
        
        print("MAE:", mae)
        print("MSE:", mse)
        print("R2 SQUARE:", r2_square)
        print("RMSE:",rmse)
        
        
        ## loging the metric
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("r2_square", r2_square)
        mlflow.log_metric("rmse",rmse)
        
        
        
        # ## logging model
        # mlflow.sklearn.log_model(model, "decision_tree_regressor")
        
        input_example = X_test.iloc[:3]
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="decision_tree_regressor",
            input_example=input_example,
            signature=signature)
        