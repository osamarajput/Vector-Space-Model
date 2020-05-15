# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:53:06 2020

@author: Osama
"""

from nltk.stem import WordNetLemmatizer
import math


#PREPROCESSING AND CLEANING


def return_stopwords(): #removal of stop words by stop list
    stopfile = open("\Stopword-list.txt","r")
    stopwords = []
    for line in stopfile:
        stopwords+=line.split()
    return stopwords

def preprocessing_list(terms):
        i=0
        lemmatizer = WordNetLemmatizer()
        stop_list=[]
        stop_list = return_stopwords()

        while i<len(terms):
            terms[i][0]=terms[i][0].casefold() #CASEFOLDING    
            terms[i][0]=terms[i][0].replace(':','') #DATA CLEANING
            terms[i][0]=terms[i][0].replace("'",'')
            terms[i][0]=terms[i][0].replace('"','')
            
            
            if(terms[i][0].find('...')>-1): #ELIMINATING TRIPLE DOTS
                tripdot_splitter=terms[i][0].split('...')
                terms[i][0]=tripdot_splitter[0]
                terms[i].append(tripdot_splitter[1])
                    
            if(terms[i][0].find(']')>-1 and terms[i][0].find('[')>-1): #ELIMINATING BRACKET SEPARATION
                terms[i][0]=terms[i][0].replace('[','')
                square_splitter=terms[i][0].split(']')
                terms[i][0]=square_splitter[0]
                terms[i].append(square_splitter[1])
                
            if(terms[i][0].find('.')>-1 and terms[i][0][-1]!='.'): #ELIMINATING DOT SEPARATION
                dot_splitter=terms[i][0].split('.')
                terms[i][0]=dot_splitter[0]
                terms[i].append(dot_splitter[1])
        
            terms[i][0]=terms[i][0].replace('.','') #DATA CLEANING
            terms[i][0]=terms[i][0].replace('?','')
            terms[i][0]=terms[i][0].replace(',','')
            
            if(terms[i][0] in stop_list): #REMOVAL OF STOPWORDS
                terms[i][0]=''
                if(terms[i][-1]!=terms[i][0] and terms[i][-1] in stop_list):
                    terms[i][-1]=''
                
            
            terms[i][0]=lemmatizer.lemmatize(terms[i][0]) #PORTER STEMMER
            if(terms[i][-1]!=terms[i][0]):
                terms[i][-1]=lemmatizer.lemmatize(terms[i][-1])


            

            i+=1
            
            
#CREATE DICTIONARY OF PREPROCESSED WORDS
            
def create_feature_dictionary():
    i=0
    feature_dict = dict()
    #f1 = open("D:\Osama\Work\Semester 6\IR\A2\checker.txt","w")

    for i in range(56):  #CREATING INDEX OF ALL DOCS
        f = open("\Trump Speechs\speech_"+str(i)+".txt","r")
        arrayy = []    
        
        
        #FILING
        for line in f:
            for word in line.split():
                arrayy+=[word.split()]
                 
        preprocessing_list(arrayy)
        for j in range(len(arrayy)):
            
            if(arrayy[j][-1]!=arrayy[j][0]):
                arrayy.append(arrayy[j][-1])
                
            
            arrayy[j]=arrayy[j][0]
            
            
        
        feature_dict[i]=arrayy
        
        
    
    return feature_dict
    

#CREATING TF AND IDF VECTORS FOR THE VOCABULARY

def create_sets():
    feature_dictionary = create_feature_dictionary()
    tf_vector = {}
    idf_vector = {}
    for i in range(56):
        tf_vector[i]={}
        feature_terms = feature_dictionary[i]
        total_terms=0
        
        #FOR TF-IDF
        for j in range(len(feature_terms)):
            if(feature_terms[j]==''):
                continue
            
            if(feature_terms[j] not in tf_vector[i].keys()):
                tf_vector[i][feature_terms[j]]=[1,0]
                #tf_vector[i][feature_terms[j]][0]=1
                total_terms+=1
            else:
                tf_vector[i][feature_terms[j]][0]+=1
                total_terms+=1
            
            if(feature_terms[j] not in idf_vector.keys()):
                idf_vector[feature_terms[j]]=[i]
                #idf_vector[feature_terms[j]][0]=1
            else:
                if(i not in idf_vector[feature_terms[j]]):
                    idf_vector[feature_terms[j]].append(i)
                
        for j in range(len(feature_terms)):
            if(feature_terms[j]==''):
                continue         
            tf_vector[i][feature_terms[j]][1]=float(tf_vector[i][feature_terms[j]][0]/total_terms)
            
            
    for key in idf_vector:
        docs = idf_vector[key]
        doc_freq = len(docs)
        idf_vector[key].clear()
        idf_vector[key].append(doc_freq)
        idf = float(56/doc_freq)
        log_idf = math.log10(idf)
        idf_vector[key].append(log_idf)
        
    for i in range(56):
        for key in tf_vector[i].keys():
            if(key in idf_vector.keys()):
                tf_value = tf_vector[i][key][1]
                idf_value = idf_vector[key][1]
                tf_vector[i][key].append(float(tf_value*idf_value))
            
        
    return_dictionary = {}
    return_dictionary[0] = tf_vector#INDEX NUMBERS AS FOLLOWS 0-tf 1-Ntf 2-product of TF-IDF
    return_dictionary[1] = idf_vector#INDEX NUMBERS AS FOLLOWS 0-df 1-log(idf)

    return return_dictionary


#PROCESS COSINE SIMILARITY
#BY PROCESSING QUERY FIRST

    

def process_query(query):
    lemmatizer = WordNetLemmatizer()
    query_splitter = query.split()
    query_vector = {}
    terms = len(query_splitter)
    vocab_vector = create_sets()
    tf_vector = vocab_vector[0]
    idf_vector = vocab_vector[1]
    
    for i in range(len(query_splitter)):
        query_splitter[i]=lemmatizer.lemmatize(query_splitter[i])
        if(query_splitter[i] not in query_vector.keys()):
            query_vector[query_splitter[i]]=[1]
        else:
            query_vector[query_splitter[i]][0]+=1
    
    for key in query_vector:
        query_vector[key].append(float(query_vector[key][0]/terms))
        if(key in idf_vector.keys()):
            idf_value = idf_vector[key][1]
            query_vector[key].append(idf_value)
            product = float(query_vector[key][1]*query_vector[key][2])
            query_vector[key].append(product)#INDEX numbers as follows 0-tf 1-Ntf 2-idf 3-product of TF-IDF 
            
    #QUERY VECTOR HAS THE PROCESSED DATA OF THE QUERY
    
    cosine_similarity = {}
    count=0
    
    for i in range(56):
        this_doc = tf_vector[i]
        dot_product = 0
        query_magnitude = 0
        doc_magnitude = 0
        magnitude_product = 0
        alpha = 0.0005
        
        
        for term in this_doc.keys():
            doc_magnitude+=float(this_doc[term][2]*this_doc[term][2])
        
        for query_term in query_vector:
            if query_term in this_doc.keys():
                dot_product+= float(query_vector[query_term][3] * this_doc[query_term][2])
                query_magnitude+= float(query_vector[query_term][3] * query_vector[query_term][3])
                #query_magnitude+ = float(pow(query_vector[query_term][3],2))
                #doc_magnitude+= float(this_doc[query_term][2] * this_doc[query_term][2])
        
        if(query_magnitude > 0 and doc_magnitude > 0):
            first_root = float(math.sqrt(query_magnitude))
            second_root = float(math.sqrt(doc_magnitude))
            magnitude_product = float(first_root*second_root)
            cosine_similarity[i]=float(dot_product/magnitude_product)
            cosine_similarity[i]=round(cosine_similarity[i],5)
            
        else:
            cosine_similarity[i]=0
            
        if(cosine_similarity[i]<alpha):
            cosine_similarity[i]=0
    
    for key in cosine_similarity:
        if(cosine_similarity[key]>0):
            count+=1
            
    #COSINE SIMILARITIES OF ALL DOCS RECORDED
    
    print('Length: ' + str(count))
    print('\n\n')
    sorted_similarities = sorted(cosine_similarity.items(), key = lambda x:(x[1],x[0]), reverse = True)
    
    #SIMILARITIES PASSED INTO A SINGLE STRING FOR GUI
    
    answer = 'The '+str(count)+' that have been retrieved are \n\n'
    
    for i in range(count):
        answer+= "\n Document number "
        answer+= str(sorted_similarities[i][0])
        answer+= " with score "
        answer+= str(sorted_similarities[i][1])
        answer+="\n"
    
    print(answer)
    return answer
        
 
 
#GRAPHICAL USER INTERFACE
 
 
import tkinter as tk
import Pmw

root = tk.Tk()

canvas1 = tk.Canvas(root, width=400, height=300, relief='raised')
canvas1.pack()


label1 = tk.Label(root, text='Vector Space Model')
label1.config(font=('arial', 14))
canvas1.create_window(200, 25, window=label1)

label2 = tk.Label(root, text='')
label2.config(font=('arial', 10))
canvas1.create_window(200, 100, window=label2)


entry1 = tk.Entry(root)
canvas1.create_window(200, 125, window=entry1)

def getResults():
    x1 = entry1.get()
    query = x1
    
    results = process_query(query)
    
    
    # Query Preprocessing
    label3 = tk.Label(root, text='The Speeches containing ' + x1 + ' are: \n \n ', font=('arial', 10))
    canvas1.create_window(200, 210, window=label3)
    
    
    top = tk.Frame(root,width=32, height=40);
    top.pack(side='top', expand=True, fill='both')
    text = Pmw.ScrolledText(top,
                        borderframe=20,
                        vscrollmode='dynamic',
                        hscrollmode='dynamic',
                        labelpos='n',
                        label_text='Speeches',
                        text_width=40,
                        text_height=4,
                        text_wrap='none',
                        )
    text.pack(expand=True)
    text.insert(tk.END,results)
    canvas1.create_window(200, 160, window=top)


button1 = tk.Button(text='Search', command=getResults, bg='brown', fg='white',
                    font=('arial', 9, 'bold'))
canvas1.create_window(200, 160, window=button1)

root.mainloop()
            
        
    
    
    