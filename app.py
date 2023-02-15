import sqlite3
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString
from data_cleaning import process_text,preprocess
from pathlib import Path
import tensorflow as tf
import pickle, re
import numpy as np
import sklearn
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

app.json_encoder = LazyJSONEncoder

app.config['JSON_SORT_KEYS'] = False

#Tools to run the function 
max_features = 100000
sentiment = ['negative', 'neutral', 'positive']
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
#===================================================================
#Load Feature Extraction for LSTM
file = open("x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

#Load Model for LSTM
model_LSTM = load_model('model_pas.h5')
#===================================================================
#Vectorizer For Neural Network
count_vect = pickle.load(open("feature.p","rb"))
#===================================================================
#Load Model for Neural Network
model_NN = pickle.load(open("model.p","rb"))
#===================================================================



#Cleansing Function
def cleansing(sent):

    string = sent.lower()

    string = re.sub(r'[^a-zA-Z0-9]',' ',string)
    string = re.sub('rt',' ',string) # Remove every retweet symbol
    string = re.sub('user',' ',string) # Remove every username
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',string) # Remove every URL
    return string



#===================================================================

#Create Swagger config and tamplate
swagger_tamplate = dict(
    info = {
        'title': LazyString(lambda: 'API Documentation for LSTM and Neural Network model'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda:'Platinum Challenge for LSTM and Neural Network model')
    },
    host = LazyString(lambda:request.host)
)

swagger_config = {
    "headers": [],
    "specs":[
        {
            "endpoint": 'docs',
            "route": '/docs.json'            
        }
    ],
    "static_url_path":"/flasgger_static",
    "swagger_ui": True,
    "specs_route":"/docs/"
}



#Initialize swagger tamplate  & config
swagger = Swagger(app, template=swagger_tamplate, config=swagger_config)



@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    return jsonify(
        info="Api data cleansing",
        status_code=200
    )
@swag_from('docs/clean.yml', methods=['POST'])
@app.route('/text_clean_form', methods=['POST'])   
def lstm_text():
    text = request.form.get('text')
    cleaned_text= cleansing(text)
    sentiment = ['negative', 'neutral', 'positive']
    predicted = tokenizer.texts_to_sequences(cleaned_text)
    guess = pad_sequences(predicted, maxlen=X.shape[1])
    prediction = model_LSTM.predict(guess)
    polarity = np.argmax(prediction[0])
    result = sentiment[polarity]

    df_new= pd.DataFrame({'raw_text':[text],'result':[result]})
    df_new.to_sql('text_cleaning',conn,if_exists='append',index=False)
    json_response= {
        'status_code': 200,
        'description': "cleaning text from user",
        'raw_data': text,
        'result': result
    }
    response_data = jsonify(json_response)
    return response_data
    #return jsonify(raw_text=text,cleaned_text=cleaned_text)




#Func API LSTM(Text)
@swag_from("docs/upload.yml", methods=['POST'])
@app.route('/Text_Processing_File', methods=['POST'])
def post_file():
    file = request.files.get('file')    
    df = pd.read_csv(file,encoding='latin-1')
    result= []
    for text in df['Tweet']:
        clean = preprocess(text)
        result.append(clean)
    raw=df['Tweet'].to_list()
    df_new1= pd.DataFrame({'raw_text':raw, 'cleaned_text':result})
    df_new1.to_sql('text_cleaning', conn, if_exists='append', index=False)
    df_new1.to_csv('preprocess_data.csv', index=False)
    response_data = jsonify(df_new1.T.to_dict())
    return response_data


if __name__ == '__main__':
    app.run(debug=True)
