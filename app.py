from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from clean import preprocess 


# load the model from disk
filename = 'nlp_model_compressed.pkl'
clf = pickle.load(gzip.open(filename,'rb'))
#clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('cv_transform.pkl','rb'))
tfidf=pickle.load(open('tfidf_transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
#   df = pd.read_csv('clean_data.csv')
#   # Features and Labels
#   df['label']=df['Satisfaction'].map({'Positive': 1, 'Negative': 0})
#   X = df['Reviews'].values.astype('U')
#   y = df['label']
#   #Extract Feature with CountVectorizer and TfidfTransformer
#   cv = CountVectorizer()
#   tfidf = TfidfTransformer()
#   X_cv = cv.fit_transform(X)
#   X_cv_tfidf = tfidf.fit_transform(X_cv)

#   pickle.dump(cv,open('cv_transform.pkl','wb'))
#   pickle.dump(tfidf,open('tfidf_transform.pkl','wb'))

#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X_cv_tfidf,y, test_size=0.25, random_state=112)

#   #Random Forest
#   from sklearn.ensemble import RandomForestClassifier
#   from sklearn.datasets import make_classification


#   RFclassifier = RandomForestClassifier()
#   RFclassifier.fit(X_train, y_train)
#   RFclassifier.score(X_test,y_test)

#   filename = 'nlp_model.pkl'
#   pickle.dump(RFclassifier, open(filename, 'wb'))

#   alternate way to create a compressed pkl file
#   joblib.dump(RFclassifier, "nlp_model_compressed.pkl", compress=9)

    if request.method == 'POST':
        message = request.form['message']
        data1 = [message]
        data = preprocess(data1)
        cv_val = cv.transform(data)
        tfidf_val = tfidf.transform(cv_val)
        my_prediction = clf.predict(tfidf_val)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)