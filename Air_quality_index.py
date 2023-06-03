import pandas as p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb

#Gathering data:
df = p.read_csv( "C:/Users/AWIKSSHIITH/OneDrive/Desktop/AQI and Lat Long of Countries.csv" )
df.dropna( inplace = True )
y = df[ 'AQI Category' ].values #Air Quality Index value and category is the target.
x = df.drop( [ 'AQI Category', 'Country', 'City' ], axis = 1 ).values #Other than Air quality index category, names of the countries and names of the cities , all other data are input data. We're not including 'AQI Category' because it is our target and 'Country', 'City' are also not included because names just don't affect the target values.
label_encoder = LabelEncoder() #Converts string data to numerical data.
x[ :, 2 ] = label_encoder.fit_transform( x[ :, 2 ] )
x[ :, 4 ] = label_encoder.fit_transform( x[ :, 4 ] )
x[ :, 6 ] = label_encoder.fit_transform( x[ :, 6 ] )
x[ :, 8 ] = label_encoder.fit_transform( x[ :, 8 ] )
y = label_encoder.fit_transform( y )

#Splitting data:
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0 )

##Training different models:
#Logistic Regression:
log_reg = LogisticRegression()
log_reg.fit( x_train, y_train )
print( 'Test accuracy of Logistic Regression is {}%.'.format( log_reg.score( x_test, y_test ) * 100 ) )
#K-nearest neighbors:
scorelist = []
for i in range( 1, 20 ):
    knn = KNeighborsClassifier( n_neighbors = i )
    knn.fit( x_train, y_train )
    scorelist.append( knn.score( x_test, y_test ) * 100 )
plt.plot( range( 1, 20 ), scorelist )
plt.title( "Plot of %Accuracy of KNN with respect to no, of neighbors" )
plt.xlabel( "No, of neighbors " )
plt.ylabel( "%Accuracy" )
plt.show()
print( 'Test accuracy of K-nearest neighbors is {} for no, of neighbors {}.'.format( max( scorelist ), scorelist.index( max( scorelist ) ) + 1 ) )
#Support Vector Machines:
svm = SVC( random_state = 1 )
svm.fit( x_train, y_train )
print( 'Test accuracy of Support Vector Machines is {}%.'.format( svm.score( x_test, y_test ) * 100 ) )
#Naive Bayes:
nb = GaussianNB()
nb.fit( x_train, y_train )
print( 'Test accuracy of Naive Bayes is {}%.'.format( nb.score( x_test, y_test ) * 100 ) )
#Decision Tree:
dt = DecisionTreeClassifier()
dt.fit( x_train, y_train )
print( 'Test accuracy of Decision Tree is {}%.'.format( dt.score( x_test, y_test ) * 100 ) )
#Random Forest:
rf = RandomForestClassifier( n_estimators = 1000, random_state = 1 )
rf.fit( x_train, y_train )
print( 'Test accuracy of Random Forest is {}%.'.format( rf.score( x_test, y_test ) * 100 ) )

#Comparing the models:
methods = [ 'Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest' ]
scores = [ log_reg.score( x_test, y_test ) * 100, max( scorelist ), svm.score( x_test, y_test ) * 100, nb.score( x_test, y_test ) * 100, dt.score( x_test, y_test ) * 100, rf.score( x_test, y_test ) * 100 ]
colors = [ 'purple', 'blue', 'green', 'yellow', 'orange', 'red' ]
plt.figure( figsize = ( 16, 5 ) )
sb.set_style( 'whitegrid' )
sb.barplot( x = methods, y = scores, palette = colors )
plt.title( "Comparing the models" )
plt.xlabel( "Methods" )
plt.ylabel( "%Accuracy" )
plt.show()

#Choosing the right model and predicting the values:
print( 'From above, we can tell that Decision Tree and Random Forest classification models have the full accuracy.' )
print( 'We can use any one of the above for predictions. I am using both.' )
y_preds_dt = dt.predict( x_test )
y_preds_rf = rf.predict( x_test )
result = p.merge( df, p.DataFrame( { 'Prediction by Decision Tree': y_preds_dt, 'Prediction by Random Forest': y_preds_rf}), left_index=True, right_index=True)
print( result )