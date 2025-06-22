from utils.data_loader import data_load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import pandas as pd 
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def run():
    X_train,X_test, y_train,y_test = data_load()
    with mlflow.start_run(run_name="RandomforestRegressor"):
        mlflow.set_tag("model_name", "RandomForestRegressor")
        random_model = RandomForestRegressor()
        random_model.fit(X_train,y_train)
        predictions = random_model.predict(X_test)
        
        
        # model evaluation
        mae=mean_absolute_error(y_test,predictions)
        mse=mean_squared_error(y_test,predictions)
        r2_square = r2_score(y_test,predictions)
        rmse=np.sqrt(mean_squared_error(y_test,predictions))
        
        
        print("MAE:",mae)
        print("MSE:", mse)
        print("R2_Square:",r2_square)
        print("RMSE:",rmse)
        
        
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("r2_sqr",r2_square)
        mlflow.log_metric("rmse", rmse)
        
        ## log the model
        
        input_example = X_test.iloc[:3]
        signature = infer_signature(X_test, random_model.predict(X_test))
        
        mlflow.sklearn.log_model(
            sk_model=random_model,
            artifact_path="RandomforestRegressor",
            input_example=input_example,
            registered_model_name="DiabetesRegressionModel",
            signature=signature)

       
        