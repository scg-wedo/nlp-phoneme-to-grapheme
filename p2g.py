#import dataframe 
import pandas as pd
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm


#import training phrase 
text = pd.read_csv('./data/on-attribute-phone_1.txt', sep= "\t", header=None, names=["sentence", "phoneme"])

#import dictionary of phoneme
dict= pd.read_csv('./dict/G2P_Dictionary_02112022 - G2P_Dictionary_02112022.csv') 
    
#unique word in domain 
def create_dic(text,dict):
    sep=[]
    for i in range(len(text['sentence'])):
        sep_text=word_tokenize(text['sentence'][i], engine="newmm")
        sep.append(sep_text)
    text['sep']=sep
    
    words=[]
    dup=0
    for i in range(len(text['sep'])):
        num=len(text['sep'][i])
        for n in range(num):
            word=text['sep'][i][n]
            if word in words:
                dup=dup+1
            else:
                words.append(word)
    #filter dictionary 
    dict_sub = dict.query("input in @words")
    return(dict_sub)

dict=create_dic(text,dict)

#clean and add new vocab
dict.drop(dict.index[dict['input']=='แบ'],inplace=True)
dict.loc[len(dict.index)+1] = ['เปิดน้ำ','p qq1 t^ n aa3 m^']
dict.loc[len(dict.index)+2] = ['เปิด','p qq1 t^']
dict.loc[len(dict.index)+3] = ['น้ำ','n aa3 m^']


dict.reset_index(drop=True,inplace=True)
#dict insert column syllable 
list=[]
for i in range(len(dict['input'])):
    A=dict['target'][i].count('^')
    list.append(A)
dict['syllable']=list


#import evaluation phoneme data
df = pd.read_csv("./data//cotto_wakeword_evaluation_new_lm.csv")
#delete null data
indexnull=df.index[df['prediction'].isnull()==True]
df.drop(index=indexnull,inplace=True)
text.drop(index=indexnull,inplace=True) #####
df.reset_index(drop=True,inplace=True)

#split phonemes and count syllable
def split_ph(df,column_name):
    list=[]
    for i in range(len(df[column_name])):
        split = df[column_name][i].split('^ ')
        for n in range(len(split)):
            split[n]=split[n].replace("^","")
        #print(split)
        list.append(split)
    df['split_'+column_name]=list
    list=[]
    for i in tqdm(range(len(df['split_'+column_name]))):#records
        len_pred = len(df['split_'+column_name][i])
        list.append(len_pred)
    df['syllable_'+column_name]=list 

split_ph(df,'prediction') # change split_pred to split_prediction & change syllable to syllable_prediction
split_ph(df,'sentence')  #change split_label to split_sentence & change split_label to syllable_sentence
split_ph(dict,'target')  #change split_dict to split_target ##dict['syllable']=dict['syllable_target']

print(df.loc[0])
print(dict.loc[0])

import kenlm
path = './model/p2g_1.arpa'
lm = kenlm.LanguageModel(path)

#mapping prediction with dict vocab
import math
import stringdist
top_k = 1
list_total_result=[]
for i in tqdm(range(len(df['split_prediction']))):#records
        output_sequences = [([], 0 , 0, 0, 0)]
        len_pred = df['syllable_prediction'][i]
        pho_pred= df['split_prediction'][i]
        list_total=[]
        for n in range(len_pred):# n syllable
                    new_sequences = []
                    new_position = 0
                    
                    for d in range(len(dict['input'])):# n dict vocab
                        len_dict=dict['syllable_target'][d]# dict_syllable
                        word=dict['input'][d]
                        pho_dict=dict['split_target'][d]
                        #print(pho_dict)
                    
                        for old_seq, old_score_lm,old_score_ph,old_total_score, old_position in output_sequences:
                            new_position = old_position + len_dict
                            if new_position <= len_pred: #still in range
                                new_seq = old_seq + [word]
                                
                                if old_score_lm ==0:
                                    score_lm = 10**(lm.score(new_seq[0]))
                                else:
                                    sentence=""
                                    for m in range(len(new_seq)):
                                        sentence = sentence + new_seq[m] + " "
                                   
                                    score_lm = 10**(lm.score(sentence))
                                score_ph = 0
                                diff = new_position-old_position
                                
                            
                                pho_pred_sub =[]
                                for p in range(old_position,new_position):
                                    pho_pred_sub.append(pho_pred[p])
                                
                                for c in range(diff):
                                    score_ph = stringdist.levenshtein_norm(pho_dict[c],pho_pred_sub[c])#1phoneme 1syllable
                                    score_ph = score_ph + score_ph
                                new_score_ph = old_score_ph + score_ph
                                new_total_score= (score_lm*200) - new_score_ph #adj
                               

                                new_sequences.append((new_seq,score_lm,new_score_ph,new_total_score,new_position))# first set 
                             
                    output_sequences = sorted(new_sequences, key = lambda val: val[3], reverse = True)
        
                        #select top-k based on score 
                        # *Note- best sequence is with the highest score
                    output_sequences = output_sequences[:top_k]
                    #print(output_sequences)            
                    #print('------------------------------------------------------------')  
                    for old_seq, old_score_lm,old_score_ph,old_total_score, old_position in output_sequences:
                        if  old_position == len_pred:                                         
                            list_total.append((old_seq,old_total_score,i)) 
        list_total_sort = sorted(list_total, key = lambda val: val[1], reverse = True)
        list_total_result.append(list_total_sort[:1])


print(len(list_total_result))
print(list_total_result[0])