#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[74]:


df = pd.read_csv(r"C:\Users\Dell\Downloads\BigML_Dataset.csv",encoding='latin')
df.head()


# In[6]:


df.drop(['url','save','publisher','description','title','author','language'], axis='columns', inplace=True)
df.head()


# In[7]:


len(df[df['stars']==5])


# In[8]:


df['dummy'] = df['stars'].apply(lambda x: 1 if x==5 else 0)
df.head()


# In[9]:


df = pd.get_dummies(df)
df.head()


# In[10]:


df.drop(['text_to_speech_Not enabled','x_ray_Not Enabled','lending_Not Enabled'], axis='columns', inplace=True)
df.head()


# In[11]:


df.info()


# In[12]:


df.describe(include='all').T


# In[13]:


df.shape


# In[14]:


plt.figure(figsize=(20, 12))
sns.pairplot(df,x_vars=['price','pages','size','customer_reviews','stars'],y_vars=['dummy'])


# In[15]:


df.hist()
plt.tight_layout()
plt.show()


# In[16]:


plt.figure(figsize=(10,15))
sns.boxplot(data=df,y='customer_reviews')


# In[17]:


plt.figure(figsize=(10,15))
sns.boxplot(data=df,y='size')


# In[18]:


plt.figure(figsize=(10,15))
sns.boxplot(data=df,y='price')


# In[19]:


plt.figure(figsize=(10,15))
sns.boxplot(data=df,y='pages')


# In[20]:


corr=df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,7))
sns.heatmap(corr, mask = mask, annot=True, center=0, cmap="YlGnBu")


# In[21]:


df.columns


# In[11]:


train,test = train_test_split(df, test_size=0.2,stratify = df['dummy'])
X_cols = ['price', 'pages', 'size', 'customer_reviews', 'text_to_speech_Enabled', 'x_ray_Enabled', 'lending_Enabled']
X_train, X_test = train[X_cols], test[X_cols]
y_train, y_test = train['dummy'], test['dummy']
num_cols = ['price', 'pages', 'size', 'customer_reviews']
train_median = X_train[num_cols].median()
X_train[num_cols]=X_train[num_cols].fillna(train_median)
X_test[num_cols]=X_test[num_cols].fillna(train_median)


# In[23]:


regs = [RandomForestClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier()]
for reg in regs:
    print("Regressão: ", reg.__class__.__name__)
    reg = reg
    reg.fit(X_train, y_train)
    print("Score: ", reg.score(X_test, y_test))
    y_proba_test = reg.predict_proba(X_test)
    roc_test = roc_auc_score(y_test, y_proba_test[:, 1])
    print("ROC AUC: ", roc_test)
    print("="*80)


# In[24]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['Atributo', 'Importancia'])
clf.sort_values('Importancia', ascending=False).head()


# In[25]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['Atributo', 'Importancia'])
clf.sort_values('Importancia', ascending=False).head()


# In[27]:


from sklearn import tree
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train) 
y_proba = clf.predict_proba(X_test)
roc_auc=roc_auc_score(y_test, y_proba[:, 1])

print(f'ROC AUC de Teste é {clf.__class__.__name__} é {roc_auc*100:.2f}%')

plt.figure(figsize=(20,10))
tree.plot_tree(clf,feature_names=X_train.columns,filled=True)


# In[28]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[29]:


reg = GaussianNB()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[23]:


pr>0.05


# In[24]:


threshold = 0.05
reg = LogisticRegression()
reg.fit(X_train, y_train)
#y_pred=reg.predict(X_test)
y_proba_test = reg.predict_proba(X_test)
y_proba_test= y_proba_test[:, 1]
y_pred = y_proba_test>threshold
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))
y_proba_test = reg.predict_proba(X_test)
roc_test = roc_auc_score(y_test, y_proba_test[:, 1])
print(roc_test)


# In[13]:


plt.hist(y_proba_test[:, 1])


# In[18]:


pr = y_proba_test[:, 1]
plt.hist(pr[y_test.values])


# In[21]:


y_train.value_counts()


# In[31]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[ ]:


=================================================================================================================================


# In[86]:


df = pd.read_csv(r"C:\Users\Dell\Downloads\BigML_Dataset.csv",encoding='latin')
df.head()


# In[87]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
words = vectorizer.fit_transform(df['description'])
svd = TruncatedSVD(n_components = 2)
lsa = svd.fit_transform(words)


# In[33]:


dictionary = vectorizer.get_feature_names()
dictionary[:5]


# In[88]:


encoding_matrix = pd.DataFrame(svd.components_,index=['Topic 1','Topic 2']).T
encoding_matrix["terms"] = dictionary
encoding_matrix['abs_topic_1'] = np.abs(encoding_matrix['Topic 1'])
encoding_matrix['abs_topic_2'] = np.abs(encoding_matrix['Topic 2'])
for i in ['0','1','2','3','4','5','6','7','8','9','è','ä','é','ã','ç','â','æ','ï','å','book','amazon','kindle']:
    encoding_matrix = encoding_matrix[~encoding_matrix.terms.str.contains(i)]
encoding_matrix.sort_values('abs_topic_1',ascending=False)


# In[77]:


encoding_matrix.sort_values('abs_topic_2',ascending=False)


# In[89]:


top_encoded_df = pd.DataFrame(lsa,columns = ['abs_topic_1','abs_topic_2'])
top_encoded_df['description']=df['description']
df['abs_topic_1']=top_encoded_df['abs_topic_1']
df['abs_topic_2']=top_encoded_df['abs_topic_2']
df.head()


# In[90]:


df['dummy'] = df['stars'].apply(lambda x: 1 if x==5 else 0)
df.drop(['url','save','publisher','description','title','author','language'], axis='columns', inplace=True)
df = pd.get_dummies(df)
df.drop(['text_to_speech_Not enabled','x_ray_Not Enabled','lending_Not Enabled'], axis='columns', inplace=True)
df.head()


# In[40]:


corr=df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,7))
sns.heatmap(corr, mask = mask, annot=True, center=0, cmap="YlGnBu")


# In[80]:


train,test = train_test_split(df, test_size=0.2,stratify = df['dummy'])
X_cols = ['price', 'pages', 'size', 'customer_reviews', 'text_to_speech_Enabled', 'x_ray_Enabled', 'lending_Enabled','abs_topic_1','abs_topic_2']
X_train, X_test = train[X_cols], test[X_cols]
y_train, y_test = train['dummy'], test['dummy']
num_cols = ['price', 'pages', 'size', 'customer_reviews','abs_topic_1','abs_topic_2']
train_median = X_train[num_cols].median()
X_train[num_cols]=X_train[num_cols].fillna(train_median)
X_test[num_cols]=X_test[num_cols].fillna(train_median)


# In[81]:


regs = [RandomForestClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier()]
for reg in regs:
    print("Regressão: ", reg.__class__.__name__)
    reg = reg
    reg.fit(X_train, y_train)
    print("Score: ", reg.score(X_test, y_test))
    y_proba_test = reg.predict_proba(X_test)
    roc_test = roc_auc_score(y_test, y_proba_test[:, 1])
    print("ROC AUC: ", roc_test)
    print("="*80)


# In[44]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['column', 'relevance'])
clf.sort_values('relevance', ascending=False).head()


# In[45]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['column', 'relevance'])
clf.sort_values('relevance', ascending=False).head()


# In[46]:


from sklearn import tree
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train) 
y_proba = clf.predict_proba(X_test)
roc_auc=roc_auc_score(y_test, y_proba[:, 1])

print(f'ROC AUC de Teste é {clf.__class__.__name__} é {roc_auc*100:.2f}%')

plt.figure(figsize=(20,10))
tree.plot_tree(clf,feature_names=X_train.columns,filled=True)


# In[47]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[48]:


reg = GaussianNB()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[49]:


reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[50]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[ ]:


================================================================================================================================


# In[91]:


df


# In[92]:


df['price']=np.log(df['price'])
df['pages']=np.log(df['pages'])
df['size']=np.log(df['size'])
df['customer_reviews']=np.log(df['customer_reviews'])
df


# In[101]:


df1 = df.sort_values(by='customer_reviews', ascending=False)
df1


# In[128]:


df=df.replace([np.inf,-np.inf], np.nan)
df


# In[129]:


print(len(np.isinf(df)[np.isinf(df)['price']==True]))
print(len(np.isinf(df)[np.isinf(df)['pages']==True]))
print(len(np.isinf(df)[np.isinf(df)['size']==True]))
print(len(np.isinf(df)[np.isinf(df)['customer_reviews']==True]))


# In[55]:


corr=df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(12,7))
sns.heatmap(corr, mask = mask, annot=True, center=0, cmap="YlGnBu")


# In[130]:


train,test = train_test_split(df, test_size=0.2,stratify = df['dummy'])
X_cols = ['price', 'pages', 'size', 'customer_reviews', 'text_to_speech_Enabled', 'x_ray_Enabled', 'lending_Enabled','abs_topic_1','abs_topic_2']
X_train, X_test = train[X_cols], test[X_cols]
y_train, y_test = train['dummy'], test['dummy']
num_cols = ['price', 'pages', 'size', 'customer_reviews','abs_topic_1','abs_topic_2']
train_median = X_train[num_cols].median()
X_train[num_cols]=X_train[num_cols].fillna(train_median)
X_test[num_cols]=X_test[num_cols].fillna(train_median)


# In[131]:


regs = [RandomForestClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier()]
for reg in regs:
    print("Regressão: ", reg.__class__.__name__)
    reg = reg
    reg.fit(X_train, y_train)
    print("Score: ", reg.score(X_test, y_test))
    y_proba_test = reg.predict_proba(X_test)
    roc_test = roc_auc_score(y_test, y_proba_test[:, 1])
    print("ROC AUC: ", roc_test)
    print("="*80)


# In[132]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['column', 'relevance'])
clf.sort_values('relevance', ascending=False).head()


# In[133]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
clf=pd.DataFrame(list(zip(X_train.columns, reg.feature_importances_)), columns=['column', 'relevance'])
clf.sort_values('relevance', ascending=False).head()


# In[135]:


from sklearn import tree
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train) 
y_proba = clf.predict_proba(X_test)
roc_auc=roc_auc_score(y_test, y_proba[:, 1])

print(f'ROC AUC de Teste é {clf.__class__.__name__} é {roc_auc*100:.2f}%')

plt.figure(figsize=(20,10))
tree.plot_tree(clf,feature_names=X_train.columns,filled=True)


# In[136]:


reg = DecisionTreeClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[137]:


reg = GaussianNB()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[139]:


reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[138]:


reg = RandomForestClassifier()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predição( 1 ou 0)')
plt.ylabel('Real ( 1 ou 0)')
print(classification_report(y_test, y_pred))


# In[ ]:


==============================================================

