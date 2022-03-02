# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:37:20 2022

@author: yipji
"""

import glob
import pandas as pd
import numpy as np
import seaborn as sns

#Patchs to files
x_train = glob.glob("C:/Users/yipji/Offline Documents/Big Datasets/feedback-prize-2021/train/*")
y_train = pd.read_csv("C:/Users/yipji/Offline Documents/Big Datasets/feedback-prize-2021/train.csv")
label_dict = {
    'Lead' : 0,
    'Position' : 1,
    'Claim' : 2,
    'Counterclaim':3,
    'Rebuttal':4,
    'Evidence':5,
    'Concluding Statement':6,
    'Start Pad':7,
    'Middle Pad':8,
    'End Pad':9
    }

def readFile(file):
    with open(file) as f:
        lines = f.readlines()
    return lines, file[-16:-4]

data, file_id = readFile(x_train[0])

test = y_train[y_train['id']==file_id]

def getFileID(index):
    field = x_train[index][-16:-4]
    return field
    
def getBoundaries(index):
    
    file_id = getFileID(index)
    temp = y_train[y_train['id']==file_id]
    
    first_token = []
    last_token = []
    for i in temp.predictionstring:
        first_token.append(int(i.split(' ')[0]))
        last_token.append(int(i.split(' ')[-1]))
        
    
    temp.insert(len(test),'first_token',first_token)
    temp.insert(len(test)+1,'last_token', last_token)
    
    temp = temp[['id','discourse_type','first_token','last_token']]
    temp['discourse_type'] = temp.discourse_type.replace(label_dict).values
    return temp

def getBIOandCat(df):
    '''
    1: Begining
    2: Inside
    3: Other
    '''
    file_id = df.id.iloc[0]
    
    #Creat a list with the categories of the start token
    BIO_list = [2]
    cat_list = [7]
    
    #Handling the case where the first entry does not start from zero
    if df.first_token.iloc[0] != 0:
        for i in range(0,df.first_token.iloc[0],1):
            BIO_list.append(3)
            cat_list.append(8)
    
    #Handling the first n-1 lines
    for i in range(len(df)-1):        
        BIO_list.append(1) #the first token is a B
        cat_list.append(df.discourse_type.iloc[i]) # append all the tokens
        for j in range(df.first_token.iloc[i]+1,df.last_token.iloc[i]+1,1):
            BIO_list.append(2)
            cat_list.append(df.discourse_type.iloc[i])
        
        #This is to account for middle unannotated tokens which will be tagged as Other
        if df.first_token.iloc[i+1]-df.last_token.iloc[i] != 1:
            for k in range(df.last_token.iloc[i]+1,df.first_token.iloc[i+1],1):
                BIO_list.append(3)
                cat_list.append(8)
    
    #Handling the last line
    BIO_list.append(1)
    cat_list.append(df.discourse_type.iloc[-1])
    for m in range(df.first_token.iloc[-1]+1,df.last_token.iloc[-1]+1,1):
        BIO_list.append(2)
        cat_list.append(df.discourse_type.iloc[-1])
    
    #Adding the categoris for the end token
    BIO_list.append(2)
    cat_list.append(9)
    
    
    if df.last_token.iloc[-1]+3 != len(cat_list) or df.last_token.iloc[-1]+3 != len(BIO_list):
        print(file_id)
        print(len(BIO_list))
        print(len(cat_list))
        print(df.last_token.iloc[-1])
        print('something went wrong')
    
    return (file_id, BIO_list, cat_list)

def getAllTargets():
    targets = pd.DataFrame(columns = ['id','BIO','cat'])
    for i in range(len(y_train[1:100])):
        x = pd.DataFrame(getBIOandCat(getBoundaries(i)), index = ['id','BIO','cat'])
        x = x.transpose()
        targets = targets.append(x)
    return targets

def getTransitions(save=False):
    '''
    This method returns a count of all the transitions from one class to another in a table
    toggle save if you which to save the results in a csv.
    '''    

    transitions = pd.DataFrame(0,columns = label_dict.keys(), index = label_dict.keys())

    subset = y_train[['id','discourse_start','discourse_end','discourse_type']]
    subset.iloc[:,3] = subset.discourse_type.replace(label_dict).values
    
    for i in subset['id'].unique():
        text = subset[subset['id']==i]
        transitions.iloc[7,text.discourse_type.iloc[0]]+=1
        transitions.iloc[text.discourse_type.iloc[-1],9]+=1
        for j in range(len(text)-1):
            if text.iloc[j+1,1]-text.iloc[j,2]>1:
                transitions.iloc[text.iloc[j,3],8]+=1
                transitions.iloc[8,text.iloc[j,3]]+=1
            else:
                transitions.iloc[text.iloc[j,3],text.iloc[j+1,3]]+=1
    
    if save:
        transitions.to_csv('transitions.csv')
    
    sns.heatmap(transitions,annot=False)    
    return transitions   

if __name__ == '__main__':
    a = getBoundaries(0)
    x,y,z = getBIOandCat(a)
    targets = getAllTargets()

    
    # transitions = getTransitions()