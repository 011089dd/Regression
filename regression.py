#all regression algorithms

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#dataset1 = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.40, random_state = 0)

# Fitting Simple Linear Regression to the Training set
# And Predicting the Test set results
from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, Y_train)
y_pred_lr = regressor_lr.predict(X_test)

# Fitting SVR to the dataset
# And Predicting a new result
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf', gamma='auto')
regressor_svr.fit(X_train, Y_train)
#y_pred_svr = regressor.predict([[6.5]])
y_pred_svr = regressor_svr.predict(X_test)

# Fitting Decision Tree Regression to the dataset
# And Predicting a new result
from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(random_state = 0)
regressor_dtr.fit(X_train, Y_train)
#y_pred_dtr = regressor.predict([[6.5]])
y_pred_dtr = regressor_dtr.predict(X_test)

# Fitting Random Forest Regression to the dataset
# And Predicting a new result
from sklearn.ensemble import RandomForestRegressor
regressor_rfg = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_rfg.fit(X_train, Y_train)
#y_pred_rfg = regressor.predict([[6.5]])
y_pred_rfg = regressor_rfg.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor_lr.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor_lr.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
