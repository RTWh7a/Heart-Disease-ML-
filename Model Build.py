import warnings
warnings.simplefilter('ignore')

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
#Load Data
file_path=(r"C:\Users\rafaa\Downloads\heart (1).csv")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
df=pd.read_csv(file_path)
#Data info
df.isnull().sum()

df.info()
#Plot info
plt.boxplot(df) #box plot
plt.show()

plt.scatter(range(len(df)),df['chol'])
plt.legend()
plt.show()

plt.pie(df['sex'].value_counts(),labels=['Female', 'Male'], autopct='%.2f%%', wedgeprops=dict(alpha=0.8))
plt.show()

sns.countplot(x=df['cp'],data=df,order=df["cp"].value_counts().index)
plt.xlabel('Pain Type')
plt.ylabel('Total')
plt.show()

plt.pie(df['target'].value_counts(),labels=['True','False'],autopct="%.2f%%",wedgeprops=dict(alpha=0.8))
plt.show()

labels=['True','False']
labels_gender=np.array([0,1])
labels_gender2=['Male','Female']
ax=pd.crosstab(df.sex,df.target).plot(kind='bar',figsize=(8,5))
plt.xlabel('Gender (Sex)', fontfamily='sans-serif', fontweight='bold')
plt.ylabel('Total', fontfamily='sans-serif', fontweight='bold')
plt.xticks(labels_gender, labels_gender2, rotation=0)
plt.legend(labels=labels, title='$\\bf{Target}$', fontsize='8', title_fontsize='9', loc='best', frameon=True)
plt.show()

fig = pd.crosstab(df.sex, df.cp).plot(kind = 'bar', color = ['coral', 'lightskyblue', 'plum', 'khaki'])
plt.title('Type of chest pain for sex')
fig.set_xticklabels(labels=['Female', 'Male'], rotation=0)
plt.legend(['pain type 0', 'pain type 1', 'pain type 2', 'pain type 3'])
plt.show()
#Splite data
x = df.drop('target',axis=1)

y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 20)

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print()
print(y.shape)
print(y_train.shape)
print(y_test.shape)

#Scale data        
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
    
#Build model
models={
    'Logistic Regression':LogisticRegression(C=100),
    'Naive Bayes':GaussianNB(),
    'Randon Forest Classfier':RandomForestClassifier(n_estimators=20,random_state=12,max_depth=5),
    'K_Nearst Neighbors':KNeighborsClassifier(n_neighbors=10),
    'Decision Tree':DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6),
    'Support Vector Machine':SVC(kernel='rbf', C=2)
}

model_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for model_name, model in models.items():

    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred = model.predict(x_test)
    
    # Evaluate the model
    #train_accuracy = accuracy_score(x_train,y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f"Model: {model_name}")
    print("Testing Accuracy: ", test_accuracy)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1 Score: ",f1)
    print("Confusion Matrix:\n ",confusion_mat)
    print("Training set score: {:.3f}".format(model.score(x_train, y_train)))
    print("Test set score: {:.3f}".format(model.score(x_test, y_test)))

    model_list.append(model_name)
    accuracy_list.append(test_accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
     
    cm=confusion_mat
    sns.heatmap(cm,annot=True,fmt='d')
    plt.title(f"Confusion matrix and its {model_name}")
    plt.ylabel("True label")
    plt.xlabel("predict label")
    plt.show()

    print("=" * 35)
    print('\n')
max_accuracy_index = accuracy_list.index(max(accuracy_list))
print(f"The best model based on accuracy is {model_list[max_accuracy_index]} with Testing Accuracy: {accuracy_list[max_accuracy_index]}")

#plot correlation of data
correlation=df.corr()
plt.figure(figsize=(30,30))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
plt.show()

#Aplly PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(x_train)
X_test_pca = pca.transform(x_test)

print(f"Original shape: {x_train.shape}")
print(f"Reduced shape: {X_train_pca.shape}")
print(f"Explained variance: {sum(pca.explained_variance_ratio_):.3f}")

X_scaled = scaler.fit_transform(x)
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.show()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Actual class')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
x_train_kmean = kmeans.fit_transform(x_train)
x_test_kmean=kmeans.transform(x_test)
kmeans_labels = kmeans.fit_predict(X_scaled)
y_pred_kmean=kmeans.fit_predict(x_test_kmean)
print(f'\nscore of Kmean\n{classification_report(y_pred_kmean,y_test)}')

print(f"Original shape: {x_train.shape}")
print(f"Reduced shape: {x_train_kmean.shape}")

#descision tree plot
plt.figure(figsize=(20,10))
plot_tree(models['Decision Tree'], filled=True, feature_names=df.columns[:-1], class_names=['Normal', 'Abnormal'])
plt.title('Decision Tree Visualization')
plt.show()

#Learning curve plots
from sklearn.model_selection import learning_curve

for model_name,model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=5, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std=np.std(train_scores, axis=1)
    test_scores_std=np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                    train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                    test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
#hirearchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')        
plt.ylabel('Distance')
plt.show()

#Importances feature
for model_name,model in models.items():
    importance=model.feature_importances_
    indices=np.argsort(importance)[::-1]
    plt.bar(range(x.shape[1]), importance[indices], color='lightblue', align='center')
    plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
    plt.title(f'Feature Importances for {model_name}')
    plt.show()