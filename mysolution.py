import matplotlib.pyplot as plt
import numpy as np
import numpy as nr
import pandas as pd
from sklearn import linear_model, preprocessing, cross_validation 
from sklearn.cross_validation import cross_val_score
from sklearn import feature_selection, metrics
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

unchanged = pd.read_csv("diabetic_data.csv",encoding='utf-8-sig')

#################  Cleaning the data #################


#Dropping and removing irrelevant data as to retain information that will help with the objective
#We set a threshold for anomalies, i.e. if more than 30% of entries in data missing, we drop
unchanged.replace(regex=r'\?', value=np.nan, inplace=True) #converting '?' and empty space values to NaN
cleaned_data = unchanged.dropna(axis=1, thresh=len(unchanged)*0.7) #setting a threshold 


reldata = cleaned_data[["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses","diabetesMed","readmitted","number_inpatient","admission_type_id","discharge_disposition_id","admission_source_id","insulin","number_emergency","age","change"]]
clustercols = cleaned_data[["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_diagnoses","number_emergency"]]
#################  Exploring the data  #################
cols  = ['num_medications','num_lab_procedures','num_procedures','number_diagnoses','number_inpatient','number_emergency']
cols2 = ['num_medications','num_lab_procedures','num_procedures','number_diagnoses','number_emergency']
def genBoxPlot(df, colname):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    df.boxplot(column = colname, ax = ax)
    ax.set_ylabel(colname)
    plt.show()

def scatterplot(col1,col2):#lets compare eg, race vs emergency
    f, ax = plt.subplots(2)
    df_w_dummies .plot.scatter(x=col1,y=col2,ax=ax[0], title="Scatter Plot")
    f.subplots_adjust(hspace=1)
    plt.show()

#For numerical columns we will look for outliers and remove them
#Function to assign value of 1 in a new col for existence of an outlier in the numerical cols
def identify_outlier(df,col): 
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    temp = np.zeros(df.shape[0])
    for i, x in enumerate(col):
        if ((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))): 
            temp[i] = 1         
    df['outlier'] = temp


#################  Identifying and Handling Outliers in the Data  #################


for c in cols:
    identify_outlier(reldata,reldata[c])


#Dropping Outliers
reldata = reldata[reldata.outlier == 0] 
reldata.drop('outlier', axis = 1, inplace = True)
genBoxPlot(reldata,'num_medications')

for x in cols2:
    identify_outlier(clustercols,clustercols[x])

clustercols = clustercols[clustercols.outlier == 0] 
clustercols.drop('outlier', axis = 1, inplace = True)
def auto_boxplot(df, plot_cols, by):
    for col in cols:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        df.boxplot(column = col, by = by, ax = ax)
        ax.set_title('Box plots of {} bg {}'.format(col, by))
        ax.set_ylabel(col)
        plt.show()

auto_boxplot(reldata, cols, "readmitted")

def _scatter_matrix(plot_cols, df):
    from pandas.tools.plotting import scatter_matrix
    fig = plt.figure(1, figsize=(10, 10))
    fig.clf()
    ax = fig.gca()
    scatter_matrix(df[plot_cols], alpha=0.3, diagonal='kde', ax = ax)
    plt.show()
    return('Done')

#Scatter matrix for data
# _scatter_matrix(cols, reldata)

f, ax = plt.subplots(2)
reldata.plot.scatter(x="number_diagnoses",y="number_emergency",ax=ax[0], title="Diagnoses vs Emergencies")
f.subplots_adjust(hspace=1)
plt.show()

f, ax = plt.subplots(2)
reldata.plot.scatter(x="time_in_hospital",y="number_emergency",ax=ax[0], title="TIH vs Emergencies")
f.subplots_adjust(hspace=1)
plt.show()
# Generating a boxplot for each of the numerical columns 

for c in cols:
    genBoxPlot(reldata,c)

noreadd = reldata.loc[reldata['readmitted'] == 'NO']
yesreadd = reldata.loc[reldata['readmitted'] != 'NO']
gt30 = reldata.loc[reldata['readmitted'] == '>30']
lt30 = reldata.loc[reldata['readmitted'] == '<30']


# Assigning dummies to non-Numerical values 
df_w_dummies = pd.get_dummies(reldata,columns=reldata[["diabetesMed","insulin","age","change"]]) 


#################  Code for visualising race distributions  #################

cauc = cleaned_data.loc[cleaned_data['race'] == 'Caucasian']
afro = cleaned_data.loc[cleaned_data['race'] == 'AfricanAmerican']
other = cleaned_data.loc[cleaned_data['race'] == 'Other']
hisp = cleaned_data.loc[cleaned_data['race'] == 'Hispanic']
quest = cleaned_data.loc[cleaned_data['race'] == '?']
asian = cleaned_data.loc[cleaned_data['race'] == 'Asian']

labels = 'Caucasian', 'African American', 'Other', 'Hispanic','?','Asian'
sizes = [len(cauc), len(afro), len(other), len(hisp),len(quest),len(asian)]
colors = ['#ffb3ba', '#ffdfba', '#ffffba', '#baffc9','#bae1ff','violet']
explode = (0,0,1,1,1,1)
plt.pie(sizes, labels=labels,explode=explode, startangle=40,colors=colors,shadow=False,pctdistance=1.2, labeldistance=0.8, autopct='%1.0f%%')
plt.axis('equal')
plt.show()

#Visualising Gender distributions

group = cleaned_data.groupby(["gender"], as_index=False).size()
#print(group)
group.plot(kind='bar',title='Gender Data')
plt.ylabel('Number')
plt.show()


#Visualising Age distributions

age = cleaned_data.groupby(["age"], as_index=False).size()
age.plot(kind='bar', title = 'Age Data')
plt.ylabel('Number of Individuals')
plt.show()

#Visualising Time Spent in Hospital  (could be useful)

timsp = cleaned_data.groupby(["time_in_hospital"], as_index=False).size()
timsp.plot(kind='pie', title = "Time Spent in hospital")
plt.xlabel('Number of Days spent in hospital')
plt.ylabel('Number of Individuals')
plt.show()

# Visualising Readmitted data
labels = 'Readmitted >30 ', 'Readmitted <30', 'Not readmitted at all'
sizes = [len(gt30), len(lt30), len(noreadd)]
colors = ['#d88f56','#ffffba','#4f6282']
explode = (0,0,0,0,0,0)
plt.pie(sizes, startangle=40,colors=colors,shadow=False,pctdistance=0.6, autopct='%1.0f%%')
plt.legend(labels,loc=3)
plt.axis('equal')
plt.show()



#################   NORMALISING THE DATA #################

# Before doing anything we will normalise the numerical data
df_w_dummies.replace('NO',0, inplace=True)
df_w_dummies.replace('>30',1, inplace=True)
df_w_dummies.replace('<30',1, inplace=True)
df_norm = df_w_dummies.select_dtypes(include=[np.number])
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min()) 
df_norm['number_emergency']=df_norm['number_emergency'].fillna(0) #Num_Emergency becomes Nan when the DF is normalised. So for this case, we will fill with 0.


clustercols.replace('NO',0, inplace=True)
clustercols.replace('>30',1, inplace=True)
clustercols.replace('<30',1, inplace=True)
cluster_norm = clustercols.select_dtypes(include=[np.number])
cluster_norm = (cluster_norm - cluster_norm.min()) / (cluster_norm.max() - cluster_norm.min()) 
cluster_norm['number_emergency']=cluster_norm['number_emergency'].fillna(0) #Num_Emergency becomes Nan when the DF is normalised. So for this case, we will fill with 0.

################# CLUSTERING   #################


# call KMeans algo with 6 clusters
model = KMeans(n_clusters=6)
model.fit(cluster_norm)
## J score
print('J-score = ', model.inertia_)
print(' score = ', model.score(cluster_norm))
## include the labels into the data
print(model.labels_)
labels = model.labels_
md = pd.Series(labels)
cluster_norm['clust'] = md
cluster_norm.head(5)
## cluster centers 
centroids = model.cluster_centers_
print ('centroids', centroids)

#Need to drop Nan Values in Clust Column
cluster_norm = cluster_norm[np.isfinite(cluster_norm['clust'])]



# histogram of the clusters
plt.hist(cluster_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.show()
# means of the clusters
print ('clustered data', cluster_norm)
print (cluster_norm.groupby('clust').mean())
# clusters as a scatter plot 
pca_data = PCA(n_components=2).fit(cluster_norm)
pca_2d = pca_data.transform(cluster_norm)
plt.scatter(pca_2d[:,0], pca_2d[:,1])
plt.title('Clusters')
plt.show()


# ################## SELECTING FEATURES FOR THE MODEL & COEFF's + MATHEMATICAL EQ   #################


model = linear_model.LogisticRegression()
X0 = df_w_dummies.ix[:, df_w_dummies.columns != 'readmitted']
Y0 = df_w_dummies['readmitted']
selector = feature_selection.RFE(model, n_features_to_select=20, step=1)
selector = selector.fit(X0, Y0)
selected_features = df_w_dummies.ix[:, selector.support_]
selected_features.drop('readmitted', axis = 1, inplace = True)
print("Selected features:\n{}".format(',\n'.join(list(selected_features))))




################## PREDICTIVE MODEL    #################


X = selected_features
Y = df_w_dummies['readmitted']
trainX, testX, trainY, testY = cross_validation.train_test_split(
X, Y, test_size=0.3, random_state=0)
clf = linear_model.LogisticRegression()
clf.fit(trainX, trainY)
predicted = clf.predict(testX)
print("Mean hits: {}".format(np.mean(predicted==testY)))
print("Accuracy score: {}".format(metrics.accuracy_score(testY, predicted)))
scores = cross_validation.cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy', cv=8)
print("Cross validation mean scores: {}".format(scores.mean()))
print("Y-axis intercept {}".format(clf.intercept_))
print("Weight coefficients:")
print("Model score:\n {}".format(clf.score(X,Y)))
print("Intercept:\n {}".format(clf.intercept_))
print("Coefficients:\n")
for feat, coef in zip(cols, clf.coef_[0]):
    print(" {:>20}: {}".format(feat, coef))
print("Score against training data: {}".format(clf.score(trainX, trainY)))
print("Score against test data: {}".format(clf.score(testX, testY)))
################## CALCULATING THE CONFUSION MATRIX @ THRESHOLDS    #################
def classify_for_threshold(clf, testX, testY, t):
    prob_df = pd.DataFrame(clf.predict_proba(testX)[:, 1])
    prob_df['predict'] = np.where(prob_df[0]>=t, 1, 0)
    prob_df['actual'] = testY
    return pd.crosstab(prob_df['actual'], prob_df['predict'])
for t in [0.2, 0.3, 0.5, 0.1, 0.01, 0.08]:
        crosstab = classify_for_threshold(clf, testX, testY, t)
        print("Threshold {}:\n{}\n".format(t, crosstab))

################## DRAWING THE ROC CURVE    #################
prob = np.array(clf.predict_proba(testX)[:, 1])
testY += 1
fpr, sensitivity, _ = metrics.roc_curve(testY, prob, pos_label=2)
print("AUC = {}".format(metrics.auc(fpr, sensitivity)))
plt.scatter(fpr, fpr, c='b', marker='s')
plt.scatter(fpr, sensitivity, c='r', marker='o')
plt.show()


