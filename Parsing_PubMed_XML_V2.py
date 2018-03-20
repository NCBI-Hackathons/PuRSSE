# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:33:18 2018

@author: jlavinder
"""

# =============================================================================
# ======================= pull in packages ====================================
# =============================================================================

import pandas as pd
#import xml.etree.ElementTree as ET
#import csv
import re, os, nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from lxml import etree

# =============================================================================
# =============================================================================

os.chdir("C:\\Users\\jlavinder\\Documents\\\\Firm Initatives\\NCBI Hackathon\\Sample Data")

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def first(parent, expr):
    children = parent.xpath(expr)
    return children[0].text if len(children) > 0 else ''



def process_xml(file):
    root = etree.parse('test_medium_data.xml').getroot()
    global raw_docs, iteration,mesh_d_text,mesh_q_text
    iteration = 0
    mesh_d_text,mesh_q_text = '',''
    raw_docs = pd.DataFrame([['']*5]*len(root.getchildren()), columns = ['PMID','Title','Abstract','MeSH_Descriptor','MeSH_Qualifier'])
    
    for pmarticle in root.getchildren():
        
        pmid = first(pmarticle,'MedlineCitation/PMID')
        title = first(pmarticle, 'MedlineCitation/Article/ArticleTitle')
        asbtract = first(pmarticle,'MedlineCitation/Article/Abstract/AbstractText')
        
        mesh_d_text,mesh_q_text = '',''
    
        for heading in pmarticle.xpath('MedlineCitation/MeshHeadingList/MeshHeading'):
            MeSH_D = first(heading,'DescriptorName')
            mesh_d_text = mesh_d_text + "$" + MeSH_D
            
            MeSH_Q = first(heading,'QualifierName')
            mesh_q_text = mesh_q_text + "$" + MeSH_Q
        
        MeSH_D = mesh_d_text.split("$")
        MeSH_Q = mesh_q_text.split("$")
        
        raw_docs['PMID'][iteration] = pmid
        raw_docs['Title'][iteration] = title
        raw_docs['Abstract'][iteration] = asbtract
        raw_docs['MeSH_Descriptor'][iteration] = MeSH_D
        raw_docs['MeSH_Qualifier'][iteration] = MeSH_Q
        
        iteration += 1
    
    return(raw_docs)
    
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

stemmer = SnowballStemmer("english")
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list = stopword_list + ['']

def tokenize_text(txt):
    tokens = nltk.word_tokenize(txt)
    tokens = [token.strip() for token in tokens]
    return tokens

def clean(doc,remove_stopwords=True):
    #tokenize text
    doc = doc.replace('-', ' ')
    doc_text=tokenize_text(doc)  
    doc_text=[x.strip() for x in doc_text]
    
    # keep only text characters
    doc_text= [re.sub("[^a-zA-Z]","", word) for word in doc_text]
    
    # lower text and remove stop words
    words = [word.lower() for word in doc_text]
    if remove_stopwords:
        words = [w for w in words if not w in stopword_list]

    # stem words and re join 
    stems = [stemmer.stem(t) for t in words if t]
    stems = ' '.join(stems)

    return(stems)

# =============================================================================
# =============================================================================
# ======================== Pulling Documents ==================================
# =============================================================================
# =============================================================================
   
#docs = process_xml('pubmed18n0003.xml')
docs = process_xml('test_medium_data.xml')
docs['Total_Text'] = docs['Title'] + ' ' + docs['Abstract']

# =============================================================================
# ======================= Cleaning Documents ==================================
# =============================================================================

Cleaned_Line_List=[] # creates matrix / variable

for line in docs['Total_Text']:
    Cleanedline = clean(line, True)
    Cleaned_Line_List.append(Cleanedline)
    
Clean_Text = pd.DataFrame(np.array(Cleaned_Line_List), columns = ['Clean_Text'])

docs = pd.concat([docs, Clean_Text], axis=1)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================






