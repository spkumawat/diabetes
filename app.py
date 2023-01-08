from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()
# from mlxtend.plotting import plot_decision_regions
# import missingno as msno
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("forest.html")

@app.route('/predict',methods=['POST','GET'])

def predict():

    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    
    # print(int_features)
    # print(final)
    # prediction=model.predict_proba(final)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)
    # output > str(0.5)

    diabetes_df = pd.read_csv('diabetes.csv')

    diabetes_df_copy = diabetes_df.copy(deep = True)
    diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


    diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
    diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
    diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
    diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
    diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)

    #scaling
    sc_X = StandardScaler()
    X =  pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,
                                                        random_state=7)

    #Imputing zeros values in the dataset


    fill_values = SimpleImputer(missing_values=0, strategy='mean')
    X_train = fill_values.fit_transform(X_train)
    X_test = fill_values.fit_transform(X_test)

    #Building the model using DecisionTree

    from sklearn.tree import DecisionTreeClassifier

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    predictions = dtree.predict(X_test)

    # Firstly we will be using the dump() function to save the model using pickle
    saved_model = pickle.dumps(dtree)

    # Then we will be loading that saved model
    dtree_from_pickle = pickle.loads(saved_model)

    # lastly, after loading that model we will use this to make predictions
    dtree_from_pickle.predict(X_test)
    dtree.feature_importances_
    
    #Plotting feature importances
    (pd.Series(dtree.feature_importances_, index=X.columns).plot(kind='barh'))

    output = []
    output  = dtree.predict(final)

    plt.plot(dtree.feature_importances_)
    
    if output[0] == 1 :
        return render_template('forest.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output[0]))

    else:
        return render_template('forest.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output[0]),name = "")


if __name__ == '__main__':
    app.run()
