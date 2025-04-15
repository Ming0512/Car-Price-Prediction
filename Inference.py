import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("LinearRegressionModel.pkl",'rb'))

# Create input sample (just like during training)
input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                          data=np.array([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']]).reshape(1, 5))
predicted_price= model.predict(input_data)
print(f"Predicted car price: {predicted_price[0]}")
