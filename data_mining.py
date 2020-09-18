# import test
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
from sklearn.preprocessing import LabelEncoder

def printModelPerformance(dict_obj):
    # st.dataframe(pd.DataFrame.from_records([dict_obj]))
    st.write("Accuracy on test set     : {:.3f}".format(dict_obj['score']))
    st.write('AUC: %.2f' % dict_obj['auc'])
    st.write('Mjority classifier Confusion Matrix:', dict_obj['confusion_majority'])
    st.write('Mjority TN = ', dict_obj['confusion_majority'][0][0])
    st.write('Mjority FP = ', dict_obj['confusion_majority'][0][1])
    st.write('Mjority FN = ', dict_obj['confusion_majority'][1][0])
    st.write('Mjority TP = ', dict_obj['confusion_majority'][1][1])

    statistic = {}
    statistic['Precision'] = dict_obj['precision_score']
    statistic['Recall'] = dict_obj['recall_score']
    statistic['F1'] = dict_obj['f1_score']
    statistic['Accuracy'] = dict_obj['accuracy_score']
    statistic_df = pd.Series(statistic.values(),index=statistic.keys())
    statistic_df.plot(kind='bar')
    st.pyplot()
    

#initialized the required data
df_main = pd.read_csv("Laundry_Data.csv")
reverse_result = pd.read_csv("reverse_result.csv")
reverse_result = reverse_result.drop('Unnamed: 0',axis=1)
final_result = reverse_result.copy()
from collections import defaultdict
d = defaultdict(LabelEncoder)
# Encoding the variable
final_result = final_result.apply(lambda x: d[x.name].fit_transform(x))


#Start body
st.title('Laundry Data Mining (Project)')
st.header('Laundry Dataset Overview')
st.dataframe(df_main.head())
st.header('Data Visualization')
st.subheader("Number of null values")
if st.checkbox("After data preprocessing:"):    
    reverse_result.isnull().sum().plot(kind='bar',figsize=(10,5))
    st.pyplot()
    st.subheader("Sample dataset after processed:")
    st.dataframe(reverse_result.head())
else:
    df_main.isnull().sum().plot(kind='bar',figsize=(10,5))
    st.pyplot()
st.subheader("More visualization on cleaned data")
columnList = []
visualize_df = reverse_result.copy()
visualize_df['count'] = visualize_df.index
for column in visualize_df:
    if column=='Date' or column=='Time' or column=='count' or column=='No':
        continue
    else:
        columnList.append(column)
selected_count_column = st.selectbox("Choose the column that you wish to see the distribution",columnList)
visualize_df[['count',selected_count_column]].groupby(selected_count_column).agg('count').plot(kind="bar",figsize=(10,5))
st.pyplot()

st.text("Choose two columns to stack together for better visualization")
stacked_column1 = st.selectbox("Choose column 1",columnList)
stacked_column2 = st.selectbox("Choose column 2",columnList)
if stacked_column1!=stacked_column2:
    temp = visualize_df.groupby([stacked_column1, stacked_column2])[stacked_column2].count().unstack(stacked_column2)
    temp.plot(kind='barh', stacked=True)
    st.pyplot()
else:
    st.warning("Column 1 and column 2 cannot be same.")

# classification
st.header('Classification & Prediction')
st.subheader("The target y value: basket_size")
st.text("The column 'shirt_type' will be dropped as this column scores the lowest in \nBoruta algorithm, which indicates that it has not much impact on the target variable.")
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
boruta_df = final_result.copy()
y = boruta_df['Basket_Size']
X = boruta_df.drop(['Date','Time','Basket_Size','shirt_type'],axis=1) #Boruta in jupyter ady showed this feature has no impact

#Choose to SMOTE or not SMOTE
if st.checkbox("With SMOTE to oversample the data: "):
    from imblearn import  over_sampling
    smt = over_sampling.SMOTE(sampling_strategy="minority", random_state=10, k_neighbors=5)
    X_res, y_res = smt.fit_resample(X, y)
else:
    X_res = X
    y_res = y
st.text("Bar chart below shows the count of target value y(basket_size) in dataset")
y_res.value_counts().plot(kind='bar',figsize=(10,5),xlabel="Basket_Size (0 is big, 1 is small)",ylabel="count")
st.pyplot()

st.subheader("Decision Tree Classifier")
#Decision Tree Classifier
dt_performance = {}
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=10)
dt=None
if st.checkbox("Tuned the parameters",key="tune_param_1"):
    criterion = st.selectbox("Choose criterion: ",['gini', 'entropy'])
    max_depth = st.selectbox("Choose max depth: ",[2,4,6,8,10,12])
    dt = DecisionTreeClassifier(criterion=criterion,random_state=10,max_depth=max_depth)
else:
    dt = DecisionTreeClassifier(random_state=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
prob_DTC = dt.predict_proba(X_test)
prob_DTC = prob_DTC[:, 1]
auc_DT= roc_auc_score(y_test, prob_DTC)
confusion_majority=confusion_matrix(y_test, y_pred)
dt_performance['auc'] = auc_DT
dt_performance['score'] = dt.score(X_test, y_test)
dt_performance['confusion_majority'] = confusion_majority
dt_performance['precision_score'] = precision_score(y_test, y_pred)
dt_performance['recall_score'] = recall_score(y_test, y_pred)
dt_performance['f1_score'] = f1_score(y_test, y_pred)
dt_performance['accuracy_score'] = accuracy_score(y_test, y_pred)
printModelPerformance(dt_performance)


st.subheader("Random Forest Classifier")
#Random forest
rf=None
if st.checkbox("Tuned the parameters",key="tune_param_2"):
    max_depth = st.selectbox("Choose max_depth:",[80, 90, 100, 110])
    max_features = st.selectbox("Choose max_features: ",[2, 3])
    min_samples_leaf = st.selectbox("Choose min_samples_leaf: ",[3, 4, 5])
    min_samples_split = st.selectbox("Choose min_samples_split: ",[8, 10, 12])
    n_estimators = st.selectbox("Choose n_estimators: ",[100, 200, 300, 1000])
    rf = RandomForestClassifier(max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,n_estimators=n_estimators)
else:
    rf = RandomForestClassifier(random_state=10)
rf_performance = {}
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
prob_RFC = rf.predict_proba(X_test)
prob_RFC = prob_RFC[:, 1]
auc_RF= roc_auc_score(y_test, prob_RFC)
confusion_majority=confusion_matrix(y_test, y_pred)
rf_performance['score'] = rf.score(X_test, y_test)
rf_performance['auc'] = auc_RF
rf_performance['confusion_majority'] = confusion_majority
rf_performance['precision_score'] = precision_score(y_test, y_pred)
rf_performance['recall_score'] = recall_score(y_test, y_pred)
rf_performance['f1_score'] = f1_score(y_test, y_pred)
rf_performance['accuracy_score'] = accuracy_score(y_test, y_pred)
printModelPerformance(rf_performance)

st.subheader("Support Vector Machine Classifier")
#SVM
svm_performance = {}
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=10)

svm=None
if st.checkbox("Tuned the parameters",key="tune_param_3"):
    kernel = st.selectbox("Choose kernel: ",['rbf','sigmoid','linear'])
    C = st.selectbox("Choose C:",[0.001, 0.01, 0.1, 1, 10])
    gamma = st.selectbox("Choose gamma: ",[0.001, 0.01, 0.1, 1])
    svm = SVC(kernel=kernel,C=C,gamma = gamma,probability=True)
else:
    svm = SVC(kernel="linear",gamma="auto", probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
prob_SVM = svm.predict_proba(X_test)
prob_SVM = prob_SVM[:, 1]
auc_SVM= roc_auc_score(y_test, prob_SVM)
confusion_majority=confusion_matrix(y_test, y_pred)
svm_performance['auc'] = auc_SVM
svm_performance['score'] = svm.score(X_test, y_test)
svm_performance['confusion_majority'] = confusion_majority
svm_performance['precision_score'] = precision_score(y_test, y_pred)
svm_performance['recall_score'] = recall_score(y_test, y_pred)
svm_performance['f1_score'] = f1_score(y_test, y_pred)
svm_performance['accuracy_score'] = accuracy_score(y_test, y_pred)
printModelPerformance(svm_performance)

st.subheader("ROC Curve to show the performance of each classifier")
#Plot the performance
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DTC) 
fpr_RFC, tpr_RFC, thresholds_RFC = roc_curve(y_test, prob_RFC) 
fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, prob_SVM) 
plt.plot(fpr_DT, tpr_DT, color='orange', label='DT') 
plt.plot(fpr_RFC, tpr_RFC, color='blue', label='RF')  
plt.plot(fpr_SVM, tpr_SVM, color='red', label='SVM')  
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()


model_list = {
    "Support Vector Machine" : svm,
    "Decision Tree":dt,
    "Random Forest":rf
}
st.header("Predict the Busket_Size by customers characteristic using model created above.")
modelInput = st.radio("Choose model to predict.",tuple(model_list.keys()))
modelSelected = model_list[modelInput]
inputList = {}
except_list = ['Day','Time_Range','Basket_Size','Time','Date','shirt_type']
inputList['Date'] = st.date_input("Input the date: ")
inputList['Time'] = st.time_input("Input the time: ")
# inputList['Age_Range'] = st.slider("Input the Age_Range: ",1,100)
for column in reverse_result:
    if column in except_list:
        continue
    inputList[column] = st.selectbox("Input the "+str(column) + ": ",reverse_result[column].unique())
if st.button("Predict"):
    inputList['Day']=calendar.day_name[inputList['Date'].weekday()]
    inputList['Time_Range'] = inputList['Time'].hour
    inputList['Date'] = df_main['Date'].iloc[0] #this column is useless
    inputList['Time'] = 21 #this column is useless
    new_df = pd.DataFrame(inputList,index=[0])
    new_df['Time_Range']=pd.cut(new_df['Time_Range'],bins = [-1,7,12,19,24],labels=['midnight','morning','afternoon','night'])
    transformed_bck = new_df.apply(lambda x: d[x.name].transform(x))
    transformed_bck = transformed_bck.drop(['Time','Date'],axis=1)
    predict_result = modelSelected.predict(transformed_bck)
    basket = None
    if predict_result[0]== 0:
        basket = "Big Basket"
    else:
        basket = "Small Basket"
    st.success("Done prediction. This customer is most likely to bring a "+str(basket))

