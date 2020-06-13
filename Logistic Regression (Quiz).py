#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/kidney_disease.csv')
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.sample(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().any()


# In[7]:


fig = plt.figure(figsize = (20,20))
sns.heatmap(df.isnull(), cbar = False, cmap = 'Accent')


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[8]:


df.info()


# In[9]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[10]:


sns.pairplot(df[['age','sod']])


# In[11]:


sns.pairplot(df[['age','sc']])


# In[12]:


sns.pairplot(df[['bgr','bu']])


# In[13]:


sns.pairplot(df[['age','bp']])


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[14]:


sns.distplot(df['age'], kde = False, bins = 20)


# In[15]:


sns.distplot(df['sod'], kde = False, bins = 5)


# In[16]:


sns.distplot(df['pot'], kde = False, bins = 10)


# In[17]:


sns.distplot(df['sc'], kde = False, bins = 5)


# In[18]:


sns.distplot(df['hemo'], kde = False, bins = 20)


# In[19]:


sns.distplot(df['bp'], kde = False, bins = 10)


# In[20]:


sns.distplot(df['sg'], kde = False, bins = 5)


# In[21]:


sns.distplot(df['bgr'], kde = False, bins = 20)


# In[22]:


sns.distplot(df['bu'], kde = False, bins = 20)


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[23]:


fig = plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), annot = df.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[24]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df, x = 'bgr', y ='su')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[25]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df, x = 'bu', y = 'hemo')


# 10. สร้าง histogram ของ feature ต่างๆที่สนใจ

# In[26]:


plt.hist(df['age'])


# In[27]:


plt.hist(df['hemo'], bins = 11, color = 'green')


# 11. สร้าง box plot ของ features ที่สนใจ

# In[28]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(df['age'], orient = 'v')


# In[29]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(df['sod'], orient = 'v')


# In[30]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(df['bp'], orient = 'v')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[31]:


sns.countplot(data = df, x = 'classification')    #ckd majority yes 


# In[32]:


sns.countplot(data = df, x = 'dm')    #diabete  majority no 


# In[33]:


sns.countplot(data = df, x = 'cad')    #coronary artery (blood vessels) majority no 


# In[34]:


sns.countplot(data = df, x = 'appet')   #appetite majority good 


# In[35]:


sns.countplot(data = df, x = 'pe')    #ขาบวม majority no 


# In[36]:


sns.countplot(data = df, x = 'ane')  #โลหิตจาง majority no 


# In[37]:


sns.countplot(data = df, x = 'rbc')      #red blood cell majority normal 


# In[38]:


sns.countplot(data = df, x = 'pc')          #pus cell majority normal        


# In[39]:


sns.countplot(data = df, x = 'pcc')   #pus cell clump majority not present   


# In[40]:


sns.countplot(data = df, x = 'ba')   #bacteria majority not present   


# In[41]:


sns.countplot(data = df, x = 'htn')


# 14. ทำ Data Cleaning โดยการลบ หรือ fill average ขึ้นอยู่กับความเหมาะสม

# In[42]:


fig = plt.figure(figsize = (20,20))
sns.heatmap(df.isnull(), cbar = False, cmap = 'Accent')


# In[43]:


age = df['age'].mean()
bp = df['bp'].mean() 
sg = df['sg'].mean()
al = df['al'].mean()
su = df['su'].mean()
bgr = df['bgr'].mean()
bu = df['bu'].mean()
sc = df['sc'].mean()
sod = df['sod'].mean()
pot = df['pot'].mean()
hemo = df['hemo'].mean()      #data int ,float use avg values to replace missing ones 


# In[44]:


df['age'].fillna(value = age, inplace = True)
df['bp'].fillna(value = bp, inplace = True)
df['sg'].fillna(value = sg, inplace = True)
df['al'].fillna(value = al, inplace = True)
df['su'].fillna(value = su, inplace = True)
df['bgr'].fillna(value = su, inplace = True)
df['bu'].fillna(value = su, inplace = True)
df['sc'].fillna(value = su, inplace = True)
df['sod'].fillna(value = su, inplace = True)
df['pot'].fillna(value = su, inplace = True)
df['hemo'].fillna(value = su, inplace = True)


# In[45]:


df['pcc'].value_counts()
df['pcc'].fillna(value = 'notpresent', inplace = True)  #replace na with majority 


# In[46]:


df['ba'].value_counts()
df['ba'].fillna(value = 'notpresent', inplace = True)  


# In[47]:


df['htn'].value_counts()
df['htn'].fillna(value = 'no', inplace = True)


# In[48]:


df['dm'].value_counts()
df['dm'].replace({'\tno': 'no', '\tyes': 'yes',' yes': 'yes'}, inplace=True)
df['dm'].fillna(value = 'no', inplace = True)


# In[49]:


df['cad'].value_counts()
df['cad'].replace({'\tno': 'no'}, inplace=True)
df['cad'].fillna(value = 'no', inplace = True)


# In[50]:


df['appet'].value_counts()
df['appet'].fillna(value = 'good', inplace = True)


# In[51]:


df['pe'].value_counts()
df['pe'].fillna(value = 'no', inplace = True)


# In[52]:


df['ane'].value_counts()
df['ane'].fillna(value = 'no', inplace = True)


# In[53]:


df.drop('pcv', axis = 1, inplace = True)
df.drop('wc', axis = 1, inplace = True)
df.drop('rc', axis = 1, inplace = True)
df.drop('rbc', axis = 1, inplace = True)
df.drop('pc', axis = 1, inplace = True)


# In[54]:


df.info()


# In[55]:


df.replace({'ckd': 1, 'notckd': 0, 'ckd\t': 1},inplace = True)    #classification  change to 1,0 


# In[56]:


df['classification'] = df['classification'].apply(int)            #classification change data type 


# In[57]:


Pcc = pd.get_dummies(df['pcc'], drop_first = True)                #dummy for object 
Pcc.rename(columns={"present": "Pcc"}, inplace = True)


# In[58]:


Ba = pd.get_dummies(df['ba'], drop_first = True)                #dummy for object 
Ba.rename(columns={"present": "Ba"}, inplace = True)


# In[59]:


Htn = pd.get_dummies(df['htn'], drop_first = True) 
Htn.rename(columns={"yes": "Htn"}, inplace = True)
Dm = pd.get_dummies(df['dm'], drop_first = True) 
Dm.rename(columns={"yes": "Dm"}, inplace = True)
Cad = pd.get_dummies(df['cad'], drop_first = True)
Cad.rename(columns={"yes": "Cad"}, inplace = True)
Appet = pd.get_dummies(df['appet'], drop_first = True) 
Appet.rename(columns={"poor": "Appet"}, inplace = True)
Pe = pd.get_dummies(df['pe'], drop_first = True) 
Pe.rename(columns={"yes": "Pe"}, inplace = True)
Ane = pd.get_dummies(df['ane'], drop_first = True) 
Ane.rename(columns={"yes": "Ane"}, inplace = True)


# In[60]:


df.drop(['pcc','ba','htn','dm','cad','appet','pe','ane'], axis = 1, inplace = True)


# In[61]:


df = pd.concat([df,Pcc,Ba,Htn,Dm,Cad,Appet,Pe,Ane], axis = 1)
df


# In[62]:


fig = plt.figure(figsize = (20,8))
sns.heatmap(df.corr(), annot = df.corr())


# 12. สร้าง train/test split ของข้อมูล สามารถลองทดสอบ 70:30, 80:20, 90:10 ratio ได้ตามใจ

# 15. เลือก features ที่สนใจนำมาเทรน และ เลือก features ทั้งหมด

# 16. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, 
# F1 score, Recall, Precision แล้วดูว่าแบบ features ที่เราเลือกมา กับ แบบเลือกทุก features แบบใดให้ผลลัพธ์ที่ดีกว่า

# In[63]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score 


# In[64]:


X = df[['al','su','bu','Htn','Dm','Appet','Pe','Ane','sg','sod','hemo']]   
#choose variables based on correlation more than 0.30 and less than 0.30


# In[65]:


y = df['classification']


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state =100)


# In[67]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)


# In[68]:


predicted = logistic_regression.predict(X_test)
predicted


# In[69]:


confusion_matrix(y_test,predicted)


# In[70]:


print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# In[71]:


X1 = df.drop('classification',axis =1)


# In[72]:


y1 = df['classification']


# In[73]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size = 0.30, random_state =100)


# In[74]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X1_train,y1_train)


# In[75]:


predicted1 = logistic_regression.predict(X1_test)
predicted1


# In[76]:


confusion_matrix(y1_test,predicted1)


# In[77]:


print('accuracy score',accuracy_score(y1_test,predicted1))
print('precision score',precision_score(y1_test,predicted1))
print('recall_score',recall_score(y1_test,predicted1))
print('f1 score',f1_score(y1_test,predicted1))


# In[78]:


X2 = df[['Htn','Dm','hemo','al','sg']]  #choose variables based on correlation more than 0.50 and less than 0.50


# In[79]:


y2 = df['classification']


# In[80]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size = 0.30, random_state =100)


# In[81]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X2_train,y2_train)


# In[82]:


predicted2 = logistic_regression.predict(X2_test)
predicted2


# In[83]:


confusion_matrix(y2_test,predicted2)


# In[84]:


print('accuracy score',accuracy_score(y2_test,predicted2))
print('precision score',precision_score(y2_test,predicted2))
print('recall_score',recall_score(y2_test,predicted2))
print('f1 score',f1_score(y2_test,predicted2))


# 17. ทำ Standardize ข้อมูล features ทั้งหมดก่อนเทรนโมเดล 

# 18. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, 
# F1 score, Recall, Precision แล้วดูว่าแบบไม่ standardize กับ แบบ standardize แบบใดให้ผลลัพธ์ดีกว่า

# In[85]:


#data transformation
X = df[['al','su','bu','Htn','Dm','Appet','Pe','Ane','sg','sod','hemo']]  
y = df['classification']


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state =100)


# In[87]:


y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1) 


# In[88]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[89]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)


# In[90]:


predicted = logistic_regression.predict(X_test)
predicted


# In[91]:


confusion_matrix(y_test,predicted2)


# In[92]:


print('accuracy score',accuracy_score(y2_test,predicted2))
print('precision score',precision_score(y2_test,predicted2))
print('recall_score',recall_score(y2_test,predicted2))
print('f1 score',f1_score(y2_test,predicted2))


# In[93]:


X1 = df.drop('classification',axis =1)
y1 = df['classification']


# In[94]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size = 0.30, random_state =100)


# In[95]:


y1_train = np.array(y_train).reshape(-1,1)
y1_test = np.array(y_test).reshape(-1,1) 


# In[96]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()

X1_train = sc_X.fit_transform(X1_train)
X1_test = sc_X.fit_transform(X1_test)


# In[97]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X1_train,y1_train)


# In[98]:


predicted1 = logistic_regression.predict(X1_test)
predicted1


# In[99]:


confusion_matrix(y1_test,predicted1)


# In[100]:


print('accuracy score',accuracy_score(y1_test,predicted1))
print('precision score',precision_score(y1_test,predicted1))
print('recall_score',recall_score(y1_test,predicted1))
print('f1 score',f1_score(y1_test,predicted1))


# 19. เลือก features ที่สนใจ และทำ standardization, เทรนโมเดล, วัดผล และเปรียบเทียบกับข้อ 18

# In[114]:


X = df[['Htn','Dm','hemo','al','sg']] #choose variables based on correlation more than 0.50 or less than 0.50
y = df['classification']


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state =100)


# In[116]:


X_train


# In[117]:


y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1) 


# In[122]:


from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[123]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)


# In[124]:


predicted = logistic_regression.predict(X_test)
predicted


# In[125]:


confusion_matrix(y_test,predicted)


# In[126]:


print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# 20. สิ่งที่สำคัญที่สุดในการทำ Machine Learning คือ ความคิดสร้างสรรค์
# ดังนั้น ลองทำวิธีใดก็ได้ตามที่สอนมาทั้งหมด เพื่อให้สุดท้ายได้ผลลัพธ์ 
# accuracy, F1 Score มากที่สุดที่ทำได้

# ในข้อ 15 ใช้ X ทุกตัว ได้ accuracy score = 1 
