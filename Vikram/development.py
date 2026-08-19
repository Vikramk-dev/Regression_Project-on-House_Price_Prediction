import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
try:

    class REGRESSION:


        def __init__(self):
            self.df=pd.read_csv("data.csv")

        def clean_data(self):
            self.df["date"]=pd.to_datetime(self.df["date"])
            self.df["month"]=self.df["date"].dt.month
            self.df["day"]=self.df["date"].dt.day
            self.df["year"]=self.df["date"].dt.year
            self.df.drop("date",axis=1,inplace=True)
            print(self.df.head(5))


            self.df["city"]=self.df["city"].astype("category").cat.codes
            self.df["country"]=self.df["country"].astype("category").cat.codes
            print("End.")

        def splitting_data(self):
            self.X=self.df.iloc[:,1:]
            self.Y=self.df.iloc[:,0]


            self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
            print(self.X_train.shape,self.Y_train.shape)
            print("End..")


        def create_model(self):

            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.Y_train)
            print("End...")

        def train_Acc_Mse(self):
            train_Predicted_values=self.reg.predict(self.X_train)
            print(train_Predicted_values)


            print("_-------------------_")

            #R2 Score
            y_trian_mean=self.Y_train.mean()

            numerator=((self.Y_train - train_Predicted_values)**2).sum()
            denominator=((self.Y_train - y_trian_mean)**2).sum()
            r2_Acc=1-(numerator/denominator)
            print(r2_Acc)
            print(r2_score(self.Y_train,train_Predicted_values))
            print("_-------------------------_")
            #Mean Square Error
            MSE=((self.Y_train - train_Predicted_values)**2).sum()/len(self.Y_train)
            print(MSE)
            print(mean_squared_error(self.Y_train,train_Predicted_values))
            print("_-------------------_")
            print("End....")

        def test_Acc_Mse(self):
            #test Acc
            test_Predicted_values=self.reg.predict(self.X_test)

            y_test_mean=self.Y_test.mean()

            num=((self.Y_test - test_Predicted_values)**2).sum()
            den=((self.Y_test - y_test_mean)**2).sum()
            r2_Acc=1-(num/den)
            print(r2_Acc)
            print(r2_score(self.Y_test,test_Predicted_values))
            print("_-------------------------_")
            #Mean Square Error
            MSE=((self.Y_test - test_Predicted_values)**2).sum()/len(self.Y_test)
            print(MSE)
            print(mean_squared_error(self.Y_test,test_Predicted_values))

            print("_-------------------_")
            print("End.....")

        def coef_inter(self):
            print(self.reg.coef_)
            print("_----------_")
            print(self.reg.intercept_)
            print("_----------_")
            print("End......")


    ob=REGRESSION()
    ob.clean_data()
    ob.splitting_data()
    ob.create_model()
    ob.train_Acc_Mse()
    ob.test_Acc_Mse()
    ob.coef_inter()

    with open("Mini_Project_1.pkl","w") as f:
        pickle.dump(ob,f)
except Exception as e:
    er_ty,er_msg,er_line=sys.exc_info()
    print(er_ty,er_msg,er_line)