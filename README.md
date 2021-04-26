Dalam pembuatan model machine learning tentunya dibutuhkan data. Sekumpulan data yang digunakan dalam machine learning disebut DATASET, yang kemudian dibagi/di-split menjadi training dataset dan test dataset.

TRAINING DATASET digunakan untuk membuat/melatih model machine learning, sedangkan TEST DATASET digunakan untuk menguji performa/akurasi dari model yang telah dilatih/di-training.

Teknik atau pendekatan yang digunakan untuk membangun model disebut ALGORITHM seperti Decision Tree, K-NN, Linear Regression, Random Forest, dsb. dan output atau hasil dari proses melatih algorithm dengan suatu dataset disebut MODEL.

Umumnya dataset disajikan dalam bentuk tabel yang terdiri dari baris dan kolom. Bagian Kolom adalah FEATURE atau VARIABEL data yang dianalisa, sedangkan bagian baris adalah DATA POINT/OBSERVATION/EXAMPLE.

Hal yang menjadi target prediksi atau hal yang akan diprediksi dalam machine learning disebut LABEL/CLASS/TARGET. Dalam statistika/matematika, LABEL/CLASS/TARGET ini dinamakan dengan Dependent Variabel, dan FEATURE adalah Independent Variabel.

Machine Learning itu terbagi menjadi 2 tipe yaitu supervised dan unsupervised Learning. Jika LABEL/CLASS dari dataset sudah diketahui maka dikategorikan sebagai supervised learning, dan jika Label belum diketahui maka dikategorikan sebagai unsupervised learning,

Mengenali email sebagai spam atau bukan spam tergolong sebagai supervised learning, karena kita mengolah dataset yang berisi data point yang telah diberi LABEL ”spam” dan “not spam”. Sedangkan jika kita ingin mengelompokkan customer ke dalam beberapa segmentasi berdasarkan variabel-variabel seperti pendapatan, umur, hobi, atau jenis pekerjaan, maka tergolong sebagai unsupervised learning

Supervised learning jika LABEL dari dataset kalian berupa numerik atau kontinu variabel seperti harga, dan  jumlah penjualan, kita memilih metode REGRESI dan jika bukan numerik atau diskrit maka digunakan metode KLASIFIKASI. Untuk unsupervised learning, seperti segmentasi customer, kita menggunakan metode CLUSTERING

### Eksplorasi Data: Memahami Data dengan Statistik - Part 1 
Membuat model machine learning tidak serta-merta langsung modelling, ada tahapan sebelumnya yang penting untuk dilakukan sehingga kita menghasilkan model yang baik. Untuk penjelasan ini, kalian akan mempraktekkan langsung ya. Kita akan memanfaatkan Pandas library. Pandas cukup powerful untuk digunakan dalam menganalisa, memanipulasi dan membersihkan data

```plantuml
import pandas as pd
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
print('Shape dataset:', dataset.shape)
print('\nLima data teratas:\n', dataset.head())
print('\nInformasi dataset:')
print(dataset.info())
print('\nStatistik deskriptif:\n', dataset.describe())
```
<details>
<summary markdown="span">Output :</summary>
Shape dataset: (12330, 18)

Lima data teratas:
    Administrative  Administrative_Duration  ...  Weekend  Revenue
0             0.0                      0.0  ...    False    False
1             0.0                      0.0  ...    False    False
2             0.0                     -1.0  ...    False    False
3             0.0                      0.0  ...    False    False
4             0.0                      0.0  ...     True    False

[5 rows x 18 columns]

Informasi dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12330 entries, 0 to 12329
Data columns (total 18 columns):
Administrative             12316 non-null float64
Administrative_Duration    12316 non-null float64
Informational              12316 non-null float64
Informational_Duration     12316 non-null float64
ProductRelated             12316 non-null float64
ProductRelated_Duration    12316 non-null float64
BounceRates                12316 non-null float64
ExitRates                  12316 non-null float64
PageValues                 12330 non-null float64
SpecialDay                 12330 non-null float64
Month                      12330 non-null object
OperatingSystems           12330 non-null int64
Browser                    12330 non-null int64
Region                     12330 non-null int64
TrafficType                12330 non-null int64
VisitorType                12330 non-null object
Weekend                    12330 non-null bool
Revenue                    12330 non-null bool
dtypes: bool(2), float64(10), int64(4), object(2)
memory usage: 1.5+ MB
None

Statistik deskriptif:
        Administrative  Administrative_Duration  ...        Region   TrafficType
count    12316.000000             12316.000000  ...  12330.000000  12330.000000
mean         2.317798                80.906176  ...      3.147364      4.069586
std          3.322754               176.860432  ...      2.401591      4.025169
min          0.000000                -1.000000  ...      1.000000      1.000000
25%          0.000000                 0.000000  ...      1.000000      2.000000
50%          1.000000                 8.000000  ...      3.000000      2.000000
75%          4.000000                93.500000  ...      4.000000      4.000000
max         27.000000              3398.750000  ...      9.000000     20.000000

[8 rows x 14 columns]
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1553">Link materi : academy.dqlab.id/main/livecode/169/328/1553</a>

----

### Eksplorasi Data: Memahami Data dengan Statistik - Part 2 
 
 Sekarang coba inspeksi nilai korelasi dari fitur-fitur berikut pada dataset_corr yang telah diberikan sebelumnya

ExitRates dan BounceRates
Revenue dan PageValues
TrafficType dan Weekend

```plantuml
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
# Tugas praktek
print('\nKorelasi BounceRates-ExitRates:', dataset_corr.loc['BounceRates', 'ExitRates'])
print('\nKorelasi Revenue-PageValues:', dataset_corr.loc['Revenue', 'PageValues'])
print('\nKorelasi TrafficType-Weekend:', dataset_corr.loc['TrafficType', 'Weekend'])

```
<details>
<summary markdown="span">Output :</summary>
Korelasi BounceRates-ExitRates: 0.9134364214595944
Korelasi Revenue-PageValues: 0.49256929525114623
Korelasi TrafficType-Weekend: -0.0022212292430307825	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1555">Link materi : academy.dqlab.id/main/livecode/169/328/1555</a>

----

### Eksplorasi Data: Memahami Data dengan Visual 
 Dalam mengeksplorasi data, kita perlu untuk memahami data dengan visual, selain dengan statistik kita juga bisa melakukan eksplorasi data dalam bentuk visual. Dengan visualisasi kita dapat dengan mudah dan cepat dalam memahami data, bahkan dapat memberikan pemahaman yang lebih baik terkait hubungan setiap variabel/ features.
 
```plantuml
import matplotlib.pyplot as plt
import seaborn as sns
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize'] = (12,5)
plt.subplot(1, 2, 1)
sns.countplot(dataset['Revenue'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 20)
plt.xlabel('revenue or not', fontsize = 14)
plt.ylabel('count', fontsize = 14)
# checking the Distribution of customers on Weekend
plt.subplot(1, 2, 2)
sns.countplot(dataset['Weekend'], palette = 'inferno')
plt.title('Purchase on Weekends', fontsize = 20)
plt.xlabel('Weekend or not', fontsize = 14)
plt.ylabel('count', fontsize = 14)
plt.show()

```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download.png">
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1558">Link materi : academy.dqlab.id/main/livecode/169/328/1558</a>

----


### Tugas Praktek 

 Dalam membuat visualisasi ini aku akan menggunakan dataset['region'] untuk membuat histogram, dan berikan judul 'Distribution of Customers' pada title, 'Region Codes' sebagai label axis-x dan 'Count Users' sebagai label axis-y.                                                                                                                                                                                                                                                                                  
```plantuml
import matplotlib.pyplot as plt
# visualizing the distribution of customers around the Region
plt.hist(dataset['Region'], color = 'lightblue')
plt.title('Distribution of Customers', fontsize = 20)
plt.xlabel('Region Codes', fontsize = 14)                                 
plt.ylabel('Count Users', fontsize = 14)
plt.show()
                                    
```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (1).png">
	</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1559">Link materi : academy.dqlab.id/main/livecode/169/328/1559</a>

----

### Data Pre-processing: Handling Missing Value - Part 1 

```plantuml
#checking missing value for each feature  
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
#Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())

```
<details>
<summary markdown="span">Output :</summary>
Counting total missing value:</br>
112	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1561">Link materi : academy.dqlab.id/main/livecode/169/328/1561</a>

----

### Data Pre-processing: Handling Missing Value - Part 2 
 
```plantuml
#Drop rows with missing value   
dataset_clean = dataset.dropna()  
print('Ukuran dataset_clean:', dataset_clean.shape) 

```
<details>
<summary markdown="span">Output :</summary>
Ukuran dataset_clean: (12316, 18)
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1563">Link materi : academy.dqlab.id/main/livecode/169/328/1563</a>

----

### Data Pre-processing: Handling Missing Value - Part 3 
 “Kalau tidak dihapus, ada metode lain yang bisa dipakai?”

“Kita bisa menggunakan metode impute missing value, yaitu mengisi record yang hilang ini dengan suatu nilai. Ada berbagai teknik dalam metode imputing, mulai dari yang paling sederhana yaitu mengisi missing value dengan nilai mean, median, modus, atau nilai konstan, sampai teknik paling advance yaitu dengan menggunakan nilai yang diestimasi oleh suatu predictive model. Untuk kasus ini, kita akan menggunakan imputing sederhana yaitu menggunakan nilai rataan atau mean.
Imputing missing value sangat mudah dilakukan di Python, cukup memanfaatkan fungsi .fillna() dan .mean() dari Pandas, seperti berikut:
```plantuml
print("Before imputation:")
# Checking missing value for each feature  
print(dataset.isnull().sum())
# Counting total missing value  
print(dataset.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with mean of feature value  
dataset.fillna(dataset.mean(), inplace = True)
# Checking missing value for each feature  
print(dataset.isnull().sum())
# Counting total missing value  
print(dataset.isnull().sum().sum())

```

<details>
<summary markdown="span">Output :</summary>
Before imputation:
Administrative             14
Administrative_Duration    14
Informational              14
Informational_Duration     14
ProductRelated             14
ProductRelated_Duration    14
BounceRates                14
ExitRates                  14
PageValues                  0
SpecialDay                  0
Month                       0
OperatingSystems            0
Browser                     0
Region                      0
TrafficType                 0
VisitorType                 0
Weekend                     0
Revenue                     0
dtype: int64
112

After imputation:
Administrative             0
Administrative_Duration    0
Informational              0
Informational_Duration     0
ProductRelated             0
ProductRelated_Duration    0
BounceRates                0
ExitRates                  0
PageValues                 0
SpecialDay                 0
Month                      0
OperatingSystems           0
Browser                    0
Region                     0
TrafficType                0
VisitorType                0
Weekend                    0
Revenue                    0
dtype: int64
0	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1565">Link materi : academy.dqlab.id/main/livecode/169/328/1565</a>

----

### Tugas Praktek 
 Praktekkan metode imputing missing value dengan menggunakan nilai median.
```plantuml
import pandas as pd
dataset1 = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

print("Before imputation:")
# Checking missing value for each feature
print(dataset1.isnull().sum())
# Counting total missing value
print(dataset1.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with median of feature value
dataset1.fillna(dataset1.median(), inplace = True)
# Checking missing value for each feature
print(dataset1.isnull().sum())
# Counting total missing value
print(dataset1.isnull().sum().sum())

```
<details>
<summary markdown="span">Output :</summary>
Before imputation:
Administrative             14
Administrative_Duration    14
Informational              14
Informational_Duration     14
ProductRelated             14
ProductRelated_Duration    14
BounceRates                14
ExitRates                  14
PageValues                  0
SpecialDay                  0
Month                       0
OperatingSystems            0
Browser                     0
Region                      0
TrafficType                 0
VisitorType                 0
Weekend                     0
Revenue                     0
dtype: int64
112

After imputation:
Administrative             0
Administrative_Duration    0
Informational              0
Informational_Duration     0
ProductRelated             0
ProductRelated_Duration    0
BounceRates                0
ExitRates                  0
PageValues                 0
SpecialDay                 0
Month                      0
OperatingSystems           0
Browser                    0
Region                     0
TrafficType                0
VisitorType                0
Weekend                    0
Revenue                    0
dtype: int64
0
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1566">Link materi : academy.dqlab.id/main/livecode/169/328/1566</a>

----

### Tugas Praktek 
 
```plantuml
from sklearn.preprocessing import MinMaxScaler  
#Define MinMaxScaler as scaler  
scaler = MinMaxScaler()  
#list all the feature that need to be scaled  
scaling_column = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
#Apply fit_transfrom to scale selected feature  
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
#Cheking min and max value of the scaling_column
print(dataset[scaling_column].describe().T[['min','max']])

```
<details>
<summary markdown="span">Output :</summary>
    			 min  max
Administrative           0.0  1.0
Administrative_Duration  0.0  1.0
Informational            0.0  1.0
Informational_Duration   0.0  1.0
ProductRelated           0.0  1.0
ProductRelated_Duration  0.0  1.0
BounceRates              0.0  1.0
ExitRates                0.0  1.0
PageValues               0.0  1.0	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/1568">Link materi : academy.dqlab.id/main/livecode/169/328/1568</a>

----

### Data Pre-processing: Konversi string ke numerik 
 LabelEncoder akan mengurutkan label secara otomatis secara alfabetik, posisi/indeks dari setiap label ini digunakan sebagai nilai numeris konversi pandas objek ke numeris (dalam hal ini tipe data int). Dengan demikian kita telah membuat dataset kita menjadi dataset bernilai numeris seluruhnya yang siap digunakan untuk pemodelan dengan algoritma machine learning tertentu

```plantuml
import numpy as np
from sklearn.preprocessing import LabelEncoder
#Convert feature/column 'Month'
LE = LabelEncoder()
dataset['Month'] =  LE.fit_transform(dataset['Month'])
print(LE.classes_)
print(np.sort(dataset['Month'].unique()))
print('')

#Convert feature/column 'VisitorType'
LE = LabelEncoder()
dataset['VisitorType'] =  LE.fit_transform(dataset['VisitorType'])
print(LE.classes_)
print(np.sort(dataset['VisitorType'].unique()))


```
<details>
<summary markdown="span">Output :</summary>
[0 1 2 3 4 5 6 7 8 9]</br>
[0 1 2 3 4 5 6 7 8 9]</br>

[0 1 2]</br>
[0 1 2]
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/328/2464">Link materi : academy.dqlab.id/main/livecode/169/328/2464</a>

----

### Features & Label 
 Dalam dataset user online purchase, label target sudah diketahui, yaitu kolom Revenue yang bernilai 1 untuk user yang membeli dan 0 untuk yang tidak membeli, sehingga pemodelan yang dilakukan ini adalah klasifikasi. Nah, untuk melatih dataset menggunakan Scikit-Learn library, dataset perlu dipisahkan ke dalam Features dan Label/Target. Variabel Feature akan terdiri dari variabel yang dideklarasikan sebagai X dan [Revenue] adalah variabel Target yang dideklarasikan sebagai y. Gunakan fungsi drop() untuk menghapus kolom [Revenue] dari dataset.
```plantuml
# removing the target column Revenue from dataset and assigning to X
X = dataset.drop(['Revenue'], axis = 1)
# assigning the target column Revenue to y
y = dataset['Revenue']
# checking the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

```
<details>
<summary markdown="span">Output :</summary>
shape of x: (12330, 17)</br>
shape of y: (12330,)
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/329/1570">Link materi : academy.dqlab.id/main/livecode/169/329/1570</a>

----

### Training dan Test Dataset 
 Dataset perlu kita bagi ke dalam training dataset dan test dataset dengan perbandingan 80:20. 80% digunakan untuk training dan 20% untuk proses testing. Fungsi Training adalah melatih model untuk mengenali pola dalam data, sedangkan testing berfungsi untuk memastikan bahwa model yang telah dilatih tersebut mampu dengan baik memprediksi label dari new observation dan belum dipelajari oleh model sebelumnya. Bagi dataset ke dalam Training dan Testing dengan melanjutkan coding yang  sudah kukerjakan ini. Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  pada fungsi train_test_split( ).
```plantuml
from sklearn.model_selection import train_test_split
# splitting the X, and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# checking the shapes
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)

```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (2).png">	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/329/1571">Link materi : academy.dqlab.id/main/livecode/169/329/1571</a>

----

### Training Model: Fit 
 kita akan menggunakan Decision Tree. Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”. Kemudian menggunakan fungsi .fit() dan X_train, y_train untuk melatih classifier tersebut dengan training dataset
```plantuml
from sklearn.tree import DecisionTreeClassifier
# Call the classifier
model = DecisionTreeClassifier()
# Fit the classifier to the training data
model = model.fit(X_train, y_train)

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/329/1572">Link materi : academy.dqlab.id/main/livecode/169/329/1572</a>

----

### Training Model: Predict 
 
```plantuml
# Apply the classifier/model to the test data
y_pred = model.predict(X_test)
print(y_pred.shape)

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/329/1573">Link materi : academy.dqlab.id/main/livecode/169/329/1573</a>

----

### Evaluasi Model Performance - Part 2 
 Untuk menampilkan confusion matrix cukup menggunakan fungsi confusion_matrix() dari Scikit-Learn
```plantuml
from sklearn.metrics import confusion_matrix, classification_report

# evaluating the model
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', model.score(X_test, y_test))

# confusion matrix
print('\nConfusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# classification report
print('\nClassification report:')
cr = classification_report(y_test, y_pred)
print(cr)

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/329/1575">Link materi : academy.dqlab.id/main/livecode/169/329/1575</a>

----

### Pemodelan Permasalahan Klasifikasi dengan Logistic Regression 
 Pemodelan Logistic Regression dengan memanfaatkan Scikit-Learn sangatlah mudah. Dengan menggunakan dataset yang sama yaitu online_raw, dan setelah dataset dibagi ke dalam Training Set dan Test Set, cukup menggunakan modul linear_model dari Scikit-learn, dan memanggil fungsi LogisticRegression() yang diberi nama logreg.

Kemudian, model yang sudah ditraining ini  bisa digunakan untuk memprediksi output/label dari test dataset sekaligus mengevaluasi model performance dengan fungsi score(), confusion_matrix() dan classification_report().

```plantuml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data  
logreg = logreg.fit(X_train, y_train)
#Training Model: Predict 
y_pred = logreg.predict(X_test)

#Evaluate Model Performance
print('Training Accuracy :', model.score(X_train, y_train))  
print('Testing Accuracy :', model.score(X_test, y_test))  

# confusion matrix
print('\nConfusion matrix')  
cm = confusion_matrix(y_test, y_pred) 
print(cm)

# classification report  
print('\nClassification report')  
cr = classification_report(y_test, y_pred)
print(cr)

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/330/1580">Link materi : academy.dqlab.id/main/livecode/169/330/1580</a>

----

### Tugas Praktek 
 Dengan menggunakan dataset online_raw.csv dan diasumsikan sudah melakukan EDA dan pre-processing, aku akan membuat model machine learning dengan menggunakan decision tree :
<ol><li>
<li>Import DecisionTreeClassifier dan panggil fungsi tersebut dengan nama decision_tree</li>
<li>Split dataset ke dalam training & testing dataset dengan perbandingan 70:30, dengan random_state = 0</li>
<li>Latih model dengan training feature (X_train) dan training target (y_train) menggunakan .fit()</li>
<li>Evaluasi hasil model decision_tree yang sudah dilatih dengan testing feature (X_test) dan print nilai akurasi dari training dan testing dengan fungsi .score()</li></ol>
 
 
```plantuml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Call the classifier
decision_tree = DecisionTreeClassifier()
# Fit the classifier to the training data
decision_tree = decision_tree.fit(X_train, y_train)

# evaluating the decision_tree performance
print('Training Accuracy :', decision_tree.score(X_train, y_train))  
print('Testing Accuracy :', decision_tree.score(X_test, y_test))  

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/330/1582">Link materi : academy.dqlab.id/main/livecode/169/330/1582</a>

----

### Tugas Praktek 
  <ol>Pisahkan dataset ke dalam Feature dan Label, gunakan fungsi .drop(). Pada dataset ini, label/target adalah variabel MEDV
<li>Checking dan print jumlah data setelah Dataset pisahkan ke dalam Feature dan Label, gunakan .shape()</li>
<li>Bagi dataset ke dalam Training dan test dataset, 70% data digunakan untuk training dan 30% untuk testing, gunakan fungsi train_test_split() , dengan random_state = 0</li>
<li>Checking dan print kembali jumlah data dengan fungsi .shape()</li>
<li>Import LinearRegression dari sklearn.linear_model</li>
<li>Deklarasikan  LinearRegression regressor dengan nama reg</li>
<li>Fit regressor ke training dataset dengan .fit(), dan gunakan .predict() untuk memprediksi nilai dari testing dataset.</li></ol>
```plantuml
#load dataset
import pandas as pd
housing = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv')
#Data rescaling
from sklearn import preprocessing
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
housing[['RM','LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(housing[['RM','LSTAT','PTRATIO','MEDV']])
# getting dependent and independent variables
X = housing.drop(['MEDV'], axis = 1)
y = housing['MEDV']
# checking the shapes
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# checking the shapes
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test :', X_test.shape)
print('Shape of y_test :', y_test.shape)

##import regressor from Scikit-Learn
from sklearn.linear_model import LinearRegression
# Call the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg = reg.fit(X_train,y_train)
# Apply the regressor/model to the test data
y_pred = reg.predict(X_test)

```
<details>
<summary markdown="span">Output :</summary>
Shape of X: (489, 3)</br>
Shape of y: (489,)</br>
Shape of X_train : (342, 3)</br>
Shape of y_train : (342,)</br>
Shape of X_test : (147, 3)</br>
Shape of y_test : (147,)
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/330/1585">Link materi : academy.dqlab.id/main/livecode/169/330/1585</a>

----

### Tugas Praktek 

kita coba hitung nilai MSE, MAE, dan RMSE dari linear modelnya :
</ol><li>Import library yang digunakan: mean_squared_error, mean_absolute_error dari  sklearn.metrics dan numpy sebagai aliasnya yaitu np. Serta, import juga matplotlib.pyplot sebagai aliasnya, plt.</li>
<li>Hitung dan print nilai MSE dan RMSE dengan menggunakan argumen y_test dan y_pred, untuk rmse gunakan np.sqrt()</li>
<li>Buat scatter plot yang menggambarkan hasil prediksi (y_pred) dan harga actual (y_test)</li></ol>
```plantuml
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
#Calculating MSE, lower the value better it is. 0 means perfect prediction
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error of testing set:', mse)
#Calculating MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error of testing set:', mae)
#Calculating RMSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', rmse)
#Plotting y_test dan y_pred
plt.scatter(y_test, y_pred, c = 'green')
plt.xlabel('Price Actual')
plt.ylabel('Predicted value')
plt.title('True value vs predicted value : Linear Regression')
plt.show()

```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (3).png">
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/330/1587">Link materi : academy.dqlab.id/main/livecode/169/330/1587</a>

----

### Tugas Praktek 
 <ol><li>Import pandas sebagai aliasnya dan KMeans dari sklearn.cluster.</li>
<li>Load dataset 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv' dan beri nama dataset</li>
<li>Diasumsikan EDA dan preprocessing sudah dilakukan, selanjutnya kita memilih feature yang akan digunakan untuk membuat model yaitu annual_income dan spending_score. Assign dataset dengan feature yang sudah dipilih ke dalam 'X'. Pada dasarnya terdapat teknik khusus yang dilakukan untuk menyeleksi feature - feature (Feature Selection) mana saja yang dapat digunakan untuk machine learning modelling, karena tidak semua feature itu berguna. Beberapa feature justru bisa menyebabkan performansi model menurun. Tetapi untuk problem ini, secara default kita akan menggunakan annual_income dan spending_score.</li>
<li>Deklarasikan  KMeans( )  dengan nama cluster_model dan gunakan n_cluster = 5. n_cluster adalah argumen dari fungsi KMeans( ) yang merupakan jumlah cluster/centroid (K).  random_state = 24.</li>
<li>Gunakan fungsi .fit_predict( ) dari cluster_model pada 'X'  untuk proses clustering.</li></ol>
```plantuml
#import library
import pandas as pd  
from sklearn.cluster import KMeans 

#load dataset
dataset = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/mall_customers.csv')

#selecting features  
X = dataset[['annual_income','spending_score']]  

#Define KMeans as cluster_model  
cluster_model = KMeans(n_clusters = 5, random_state = 24)  
labels = cluster_model.fit_predict(X)

```

</br>
<a href="https://academy.dqlab.id/main/livecode/169/331/1590">Link materi : academy.dqlab.id/main/livecode/169/331/1590</a>

----

### Tugas Praktek 

```plantuml
#import library
import matplotlib.pyplot as plt

#convert dataframe to array
X = X.values
#Separate X to xs and ys --> use for chart axis
xs = X[:,0]
ys = X[:,1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = cluster_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D', s=50)
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

```
<details>
<summary markdown="span">Output :</summary>
	<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (4).png">
	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/331/1591">Link materi : academy.dqlab.id/main/livecode/169/331/1591</a>

----

### Tugas Praktek 

```plantuml
#import library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Elbow Method - Inertia plot
inertia = []
#looping the inertia calculation for each k
for k in range(1, 10):
	#Assign KMeans as cluster_model
	cluster_model = KMeans(n_clusters = k, random_state = 24)
	#Fit cluster_model to X
	cluster_model.fit(X)
	#Get the inertia value
	inertia_value = cluster_model.inertia_
	#Append the inertia_value to inertia list
	inertia.append(inertia_value)
    
##Inertia plot
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('inertia')
plt.show()

```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (5).png">
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/331/1593">Link materi : academy.dqlab.id/main/livecode/169/331/1593</a>

----

### Case Study: Promos for our e-commerce - Part 1 
Adapun feature - feature dalam dataset ini adalah :

<ol><li>'Daily Time Spent on Site' : lama waktu user mengunjungi site (menit)</li>
<li>'Age' : usia user (tahun)</li>
<li>'Area Income' : rata - rata pendapatan di daerah sekitar user</li>
<li>'Daily Internet Usage' : rata - rata waktu yang dihabiskan user di internet dalam sehari (menit)</li>
<li>'Ad Topic Line' : topik/konten dari promo banner</li>
<li>'City' : kota dimana user mengakses website</li>
<li>'Male' : apakah user adalah Pria atau bukan</li>
<li>'Country' : negara dimana user mengakses website</li>
<li>'Timestamp' : waktu saat user mengklik promo banner atau keluar dari halaman website tanpa mengklik banner</li>
<li>'Clicked on Ad' : mengindikasikan user mengklik promo banner atau tidak (0 = tidak; 1 = klik).</li></ol>
```plantuml
#import library 
import pandas as pd

# Baca data 'ecommerce_banner_promo.csv'
data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/ecommerce_banner_promo.csv')

#1. Data eksplorasi dengan head(), info(), describe(), shape
print("\n[1] Data eksplorasi dengan head(), info(), describe(), shape")
print("Lima data teratas:")
print(data.head())
print("Informasi dataset:")
print(data.info())
print("Statistik deskriptif dataset:")
print(data.describe())
print("Ukuran dataset:")
print(data.shape)

```
<details>
<summary markdown="span">Output :</summary>
Ukuran dataset:</br>
(1000, 10) 	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/1596">Link materi : academy.dqlab.id/main/livecode/169/332/1596</a>

----

### Case Study: Promos for our e-commerce - Part 2 


```plantuml
#2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
print("\n[2] Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()")
print(data.corr())

#3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
print("\n[3] Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()")
print(data.groupby('Clicked on Ad').size())

```
<details>
<summary markdown="span">Output :</summary>
Clicked on Ad</br>
0    500</br>
1    500</br>
dtype: int64	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/2466">Link materi : academy.dqlab.id/main/livecode/169/332/2466</a>

----

### Case Study: Promos for our e-commerce - Part 3  
*Jumlah user dibagi ke dalam rentang usia menggunakan histogram (hist()), gunakan bins = data.Age.nunique() sebagai argumen. nunique() adalah fungsi untuk menghitung jumlah data untuk setiap usia (Age).
*Gunakan pairplot() dari seaborn modul untuk menggambarkan hubungan setiap feature. 

```plantuml
#import library
import matplotlib.pyplot as plt
import seaborn as sns
#Seting: matplotlib and seaborn
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
#4. Data eksplorasi dengan visualisasi
#4a. Visualisasi Jumlah user dibagi ke dalam rentang usia (Age) menggunakan histogram (hist()) plot
plt.figure(figsize=(10, 5))
plt.hist(data['Age'], bins = data.Age.nunique())
plt.xlabel('Age')
plt.tight_layout()
plt.show()
#4b. Gunakan pairplot() dari seaborn (sns) modul untuk menggambarkan hubungan setiap feature.
plt.figure()
sns.pairplot(data)
plt.show()

```
<details>
<summary markdown="span">Output :</summary>
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (6).png"></br>	
<img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/download (7).png">
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/2465">Link materi : academy.dqlab.id/main/livecode/169/332/2465</a>

----

### Case Study: Promos for our e-commerce - Part 4


```plantuml
#5. Cek missing value
print("\n[5] Cek missing value")
print(data.isnull().sum().sum())

```
<details>
<summary markdown="span">Output :</summary>
[5] Cek missing value</br>
0
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/2467">Link materi : academy.dqlab.id/main/livecode/169/332/2467</a>

----

### Case Study: Promos for our e-commerce - Part 5 


```plantuml
#import library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing
print("\n[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing")
#6a.Drop Non-Numerical (object type) feature from X, as Logistic Regression can only take numbers, and also drop Target/label, assign Target Variable to y.
X = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis = 1)
y = data['Clicked on Ad']
#6b. splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#6c. Modelling
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train,y_train)
# Prediksi model
y_pred = logreg.predict(X_test)
#6d. Evaluasi Model Performance
print("Evaluasi Model Performance:")
print("Training Accuracy :", logreg.score(X_train, y_train))
print("Testing Accuracy :", logreg.score(X_test, y_test))

```
<details>
<summary markdown="span">Output :</summary>
[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing</br>
Evaluasi Model Performance:</br>
Training Accuracy : 0.9</br>
Testing Accuracy : 0.9	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/1600">Link materi : academy.dqlab.id/main/livecode/169/332/1600</a>

----

### Case Study: Promos for our e-commerce - Part 6
```plantuml
#Import library
from sklearn.metrics import confusion_matrix, classification_report

#7. Print Confusion matrix dan classification report
print("\n[7] Print Confusion matrix dan classification report")

#apply confusion_matrix function to y_test and y_pred
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#apply classification_report function to y_test and y_pred
print("Classification report:")
cr = classification_report(y_test, y_pred)
print(cr)
```
<details>
<summary markdown="span">Output :</summary>
</br>
[7] Print Confusion matrix dan classification report</br>
Confusion matrix:</br>
[[86  3]</br>
 [17 94]]</br>
Classification report:</br>
 |           | precision | recall | f1-score | support |
 |     :--   |   :---:   |  :---: |   :---:  |  :---:  |
 |         0 |    0.83   |   0.97 |   0.90   |   89    |
 |         1 |    0.97   |  0.85  |   0.90   |   111   |
 |avg / total|    0.91   |   0.90 |   0.90   |   200   |	
</details>
</br>
<a href="https://academy.dqlab.id/main/livecode/169/332/2468">Link materi : academy.dqlab.id/main/livecode/169/332/2468</a>

----

<p align="center"><b>E-Sertifikat </b></br><img src="https://github.com/yenysyafitry/DQLab-Machine-Learning-With-Python-for-Beginner/blob/main/e-sertifikat.jpg"></p>
