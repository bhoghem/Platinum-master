import pandas as pd
import re
import sqlite3

db = sqlite3.connect('challenge.db', check_same_thread=False)
db.text_factory = bytes
mycursor = db.cursor()
q_kamusalay = "select * from kamusalay"
t_kamusalay = pd.read_sql_query(q_kamusalay, db)


def filterAbusive():
    list_abusive = pd.read_csv('abusive.csv')
    list_abusive = list_abusive['ABUSIVE'].str.lower().tolist()
    return list_abusive

def filterBadWord(val):
    i = val
    i = val.lower()
    list_abusive = filterAbusive()
    for j in list_abusive:
        if j in i:
            holder = i.replace(j, '*******')
            val = holder
            i = holder
    print(val)
    return val


def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ', text)
    text = re.sub('rt',' ', text)
    text = re.sub('user',' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text

alay_dict_map = dict(zip(t_kamusalay['alay'], t_kamusalay['cleaned'])) #zip menyatukan value dengan index yang sama
def normalize_alay(text):
    for word in alay_dict_map:
        return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


# Untuk Proses Cleaning Data
def preprocess(text):
    text = lowercase(text) # 1
    text = remove_unnecessary_char(text) # 2
    text = remove_nonaplhanumeric(text) # 3
    text = normalize_alay(text)
    text = filterBadWord(text)
    return text






# Untuk Proses Text
def process_text(input_text):
    try: 
        output_text = preprocess(input_text)
        return output_text
    except:
        print("Text is unreadable")
