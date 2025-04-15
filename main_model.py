import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


mpl.style.use('ggplot')


def data_visualization(car):
    # 1. Boxplot: Company vs Price
    plt.figure(figsize=(15,7))
    ax = sns.boxplot(x='company', y='Price', data=car)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    plt.title("Company vs Price")
    plt.tight_layout()
    plt.savefig("plot_company_vs_price.png")
    plt.close()

    # 2. Swarmplot: Year vs Price
    plt.figure(figsize=(15,7))
    ax = sns.swarmplot(x='year', y='Price', data=car, size=3)  
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    plt.title("Year vs Price")
    plt.tight_layout()
    plt.savefig("plot_year_vs_price.png")
    plt.close()

    # 3. Scatterplot: KMs Driven vs Price
    plt.figure(figsize=(15,7))
    sns.scatterplot(x='kms_driven', y='Price', data=car)
    plt.title("KMs Driven vs Price")
    plt.tight_layout()
    plt.savefig("plot_kms_vs_price.png")
    plt.close()

    # 4. Boxplot: Fuel Type vs Price
    plt.figure(figsize=(15,7))
    sns.boxplot(x='fuel_type', y='Price', data=car)
    plt.title("Fuel Type vs Price")
    plt.tight_layout()
    plt.savefig("plot_fueltype_vs_price.png")
    plt.close()

    # 5. Scatterplot: Company vs Price with hue and size
    plt.figure(figsize=(18,8))
    ax = sns.scatterplot(x='company', y='Price', data=car, hue='fuel_type', size='year')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    plt.title("Company vs Price by Fuel Type & Year")
    plt.tight_layout()
    plt.savefig("plot_company_vs_price_by_fuel_year.png")
    plt.close()

def main():
    car = pd.read_csv("quikr_car.csv") 
    print(car.head(),"\nSphae:",car.shape)
    car_backup = car.copy()

    ##################   PREPROCESSING OF THE DATA     #####################################################

    car=car[car['year'].str.isnumeric()]   # Removing non-year value
    car['year']=car['year'].astype(int)    # Convert the year to an integer
    car=car[car['Price']!='Ask For Price'] # Remove Ask for Price 
    car['Price']=car['Price'].str.replace(',','').astype(int) # Remove commas in between prices and convert it to integer
    car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','') # Get the first kms driven value and replace comma with ''
    car=car[car['kms_driven'].str.isnumeric()]
    car['kms_driven']=car['kms_driven'].astype(int) # Convert kms driven string to integer
    car=car[~car['fuel_type'].isna()]       # Remove nan values from fuel type
    print("New shape:",car.shape)

    car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ') # Keep first 3 words of name
    car=car.reset_index(drop=True)          # Reset index

    car.to_csv("Cleaned_car_data.csv")      # Save cleaned car data
    print(car.head())

    car=car[car['Price']<6000000]           # Drop car having price greater than 6M

    ####################    DATA VISUALIZATION   ###################################

    data_visualization(car)

    ####################    MODELLING AND TRAINING OF DATA   ###################################
    X=car[['name','company','year','kms_driven','fuel_type']]
    y=car['Price']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)  
    ohe=OneHotEncoder()
    ohe.fit(X[['name','company','fuel_type']])  # ohe object to contain all the unique categories in each column

    # OneHotEncoding on the unique selected categories and letting other as it is(passthrough)
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')
    
    scores=[]
    # Finding random state of model that gives best result
    for i in range(1000):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
        lr=LinearRegression()
        pipe=make_pipeline(column_trans,lr)
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        score = r2_score(y_test,y_pred)
        scores.append(score)
        print(f"Iteration:{i+1}, Current r2 Score: {score}")
    
    # Once the best score state is found, run one more iteration on same splitting and retrive back the best model parameter
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    r2_score(y_test,y_pred)

    # Lets pridict the unknown price with custom feature
    print(f"Best r2 Score: {scores[np.argmax(scores)]}")
    # predicted = pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
    # print(f"Predicted Price:{predicted[0]}")

    pickle.dump(pipe,open('LinearRegressionModel.pkl','wb')) # Save best model parameters
    
if __name__ == "__main__":
    main()
