# =============================================================================
# this script is designed to perform several tasks related to the gathering, 
# cleaning, analyzing, and more gathering of Public Health journal articles
# =============================================================================

import pandas as pd
import re, os, nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from lxml import etree

# =============================================================================
# Input Variables 
# =============================================================================
home_dir="/home/ubuntu/data"  #change accordingly
target_dataset='10000urine.xml' # your large corpus to be used to search across
validation_dataset="300GoutGeneticsTreatementArticles.xml" #smaller set you want to ultimately find
pmid_list=['26810134','25676789','27798726','20962433','26086348'] #list of PubMed IDs

#TODO: develop functions to allow input (such as PubMed IDs)
os.chdir(home_dir) #set directory

#TODO: write some kind of function to automatically install and download
# nltk dependencies
nltk.download('stopwords') #required 
nltk.download('punkt') #required 

# =============================================================================
# Functions for XML Parsing
# =============================================================================
def first(parent, expr):
    children = parent.xpath(expr)
    return children[0].text if len(children) > 0 else ''

def process_xml(file):
    root = etree.parse(file).getroot()
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

# =============================================================================
# Functions for cleaning + pre-processing text data
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
    doc=doc.replace('-', ' ')
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
# Begin processing of data
# =============================================================================
#add new field
docs = process_xml(target_dataset)
docs['Total_Text'] = docs['Title'] + ' ' + docs['Abstract']

Cleaned_Doc_List=[] # creates matrix / variable

for line in docs['Total_Text']:
    Cleanedline=clean(line, True) #runs methods for each abstract in Doc_List
    Cleaned_Doc_List.append(Cleanedline) #creates new matrix of Cleaned_Doc_List
    
Clean_Text = pd.DataFrame(np.array(Cleaned_Doc_List), columns = ['Clean_Text'])

docs = pd.concat([docs, Clean_Text], axis=1)

validation_articles=process_xml(validation_dataset) # validation set 
validation_articles['Total_Text'] = validation_articles['Title'] + ' ' + validation_articles['Abstract']
clean_validation_set=[]
for line in validation_articles['Total_Text']:
    Cleanedline=clean(line, True) #runs methods for each abstract in Doc_List
    clean_validation_set.append(Cleanedline) #creates new matrix of Cleaned_Doc_List
    
clean_validation_df = pd.DataFrame(np.array(clean_validation_set), columns = ['Clean_Text'])

validation_articles= pd.concat([validation_articles, clean_validation_df], axis=1)

#concatenate the dataframes to force the validation set into 
frames=[docs,validation_articles]
all_docs=pd.concat(frames)

all_docs=all_docs.set_index('PMID')
#drop duplicates
all_docs=all_docs[~all_docs.index.duplicated(keep='first')]

#verify the IDs are in the larger corpus
flagged_gout_IDs=all_docs[all_docs.index.isin(pmid_list)]
#TODO: create a validation function to catch an error here if PMIDs are 
#not present

#transform to lists 
clean_text=all_docs['Clean_Text'].tolist()
mesh_list=all_docs['MeSH_Descriptor'].tolist()
#mesh_list_c=[[x for x in y if x]  for y in mesh_list if y] #get rid of empty strings in meSH terms

#Get the row numbers for where our docs are
row_nb_list=[]
for i, ind in enumerate(all_docs.index):
    if ind in pmid_list:
        row_nb_list.append(i)

# =============================================================================
# Topic Modeling! 
# =============================================================================

#####Steps######
#0 - Tf-Idf
#1 - LDA
#2 - w2v
#run on abstract, MeSH, keywords 
#3 - combine - the results 
#4 - run K nearest neighbors

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Abstract TF-IDF
bow_vectorizer=CountVectorizer()
bow=bow_vectorizer.fit_transform(clean_text)
word_counts=bow.toarray()
tfidf_transformer=TfidfTransformer()
tfidf=tfidf_transformer.fit_transform(word_counts)

#dimension reduction
from sklearn.decomposition import TruncatedSVD
TSVD = TruncatedSVD(n_components=200, algorithm = "randomized", n_iter = 5)
TSVD_fit=TSVD.fit(tfidf)
TSVD_reduced=TSVD.fit_transform(tfidf)

# Latent Dirchlet Allocation
from sklearn.decomposition import LatentDirichletAllocation
lda_ = LatentDirichletAllocation(n_components=50, max_iter=500,
        learning_method='online',
        learning_offset=50.,
        total_samples = len(clean_text),
        random_state=0)
lda_tx=lda_.fit_transform(word_counts) #fit transform 

#save models - especially important for LDA taking so long to run
import pickle
from sklearn.externals import joblib
#joblib.dump(lda_, 'filename.pkl')
#joblib.dump(lda_tx, 'lda_tx.pkl')
# pickle.dump(lda_,open('lda_output.txt','wb'))

m_list1=[' '.join(el) for el in mesh_list] #comes in as list of lists

#MeSH term TF IDF
Mesh_bow_vectorizer=CountVectorizer()
mesh_bow=bow_vectorizer.fit_transform(m_list1)
mesh_word_counts=bow.toarray()
mesh_tfidf_transformer=TfidfTransformer()
mesh_tfidf=tfidf_transformer.fit_transform(mesh_word_counts)

#TODO: consider making the above lines into a reusable function 

#dimension reduction for MeSH
MeSH_TSVD_reduced=TSVD.fit_transform(mesh_tfidf)

# Doc2vec for the article abstracts (document level)
from gensim.models import doc2vec
justwords = [x.split() for x in clean_text] # assumes a format of a list of tuples- 0 = country name, 1= a list of strings, 1 string for each word
justwords =np.array(justwords )
from gensim.models import doc2vec
DocIDs=[x for x in all_docs.index]
from collections import namedtuple
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
docs=[]
for i, text in enumerate(justwords):
    words=text
    tags=[i]
    docs.append(analyzedDocument(words, tags))

size=200  #specify the number of feature vectors to keep
d2v_model=doc2vec.Doc2Vec(docs,size = size, window = 10, min_count = 2, workers = 2)

#TODO: in distant future, consider allowing users to specify these parameters
#for the NLP piece 

docvec_list=[]
for i in range(len(docs)):
    docvec_list.append(d2v_model.docvecs[i])

doc_vec_df=pd.DataFrame({'wordvecs':docvec_list})
docvec_array=np.asarray(docvec_list)

#word2vec for MeSH terms
from gensim.models.word2vec import Word2Vec
X= np.array(mesh_list)
model = Word2Vec(X, size=20, window=10, min_count=2, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
model1=model

#transform model output into an easy to handle dataframe
word_order = []
word_vectors = np.array([])
#print(type(word_vectors))
for i, word in enumerate(model1.wv.index2word):
    vector_rep = model1[word]
    #print vector_rep
    word_order.append(word)
    if i ==0:
        word_vectors = np.append(word_vectors, vector_rep)
    else:
        word_vectors = np.vstack((word_vectors, vector_rep))
#    print(word,vector_rep)
wv_df= pd.DataFrame(word_vectors,index=word_order)

#Old code, keeping in case we want to use PCA for something in the future
#from sklearn.decomposition import PCA
##reduce to 2
#pca = PCA(n_components=2)
##reduced_df = pd.DataFrame(pca.fit_transform(df.values),index=word_order)
#reduced_df = pd.DataFrame(pca.fit_transform(wv_df.values),index=word_order)
#reduced_df.columns=['x','y']

#into a dataframe 
df_w2v = pd.DataFrame(word_vectors,index=word_order)
#df_w2v.index

#output
#df_w2v.to_csv('MeSH_word_embeddings.csv')
#TODO: depending on our use case, we may not want this to be an output option

# =============================================================================
# End of Feature Creation
# =============================================================================

#Now combine the feature vectors - use only doc2vec, tfidf for docs, and LDA for KNN
all_txt_features=np.hstack([TSVD_reduced,lda_tx,docvec_array])

# =============================================================================
# Modeling
# =============================================================================
#from sklearn.neighbors import KNeighborsClassifier

#Try K-D Tree instead 
from sklearn.neighbors import KDTree
# row_nb_list - use this for indexing
kdt = KDTree(all_txt_features, leaf_size=30, metric='euclidean')
#TODO: low-priority for now, consider updating to another distance metric (cosine)

#Using KDTree to find articles similar to that of our original documents
# returned_list=[]
# for el in row_nb_list:
dist, idx = kdt.query(all_txt_features,k=50)

#TODO: if we want to get fancy, since there is document overlap in the retrieved
#results, we could have a function that would dynamically run and retrieve 
# until we get the exact amount the user specifies 

#TODO: consider adding input upfront for user to specify exactly number of 
# articles they want to have retrieved 

#pickle and save 
#s = pickle.dumps(kdt)
# tree_copy = pickle.loads(s)    
#pickle.dump(s,open('KD-Ball_10000_results.pkl','wb'))

#throw K-D ball results of neighbors and distance into a DF
distance_list=[]
neighbor_list=[]
for i,el in enumerate(idx):
    if idx[i][0] in row_nb_list:
        neighbor_list.append(idx[:][i])
        distance_list.append(dist[:][i])
        
neighbord_info_df=pd.DataFrame({
    "Doc Index":row_nb_list,
    "Neighbor":neighbor_list,
    "Distance":distance_list
})

#save this
# neighbord_info_df.to_csv('K-D_Ball_5article_neighbors_distance.csv')

# =============================================================================
# End of document modeling - now retrieve the new document information
# =============================================================================

# Index-mapping-lookup rigamarole to get the documents 
doc_list=neighbord_info_df.Neighbor.tolist()
doc_list=[item for sublist in doc_list for item in sublist]
doc_list=list(set(doc_list))
doc_list=[str(el) for el in doc_list]
# translate position to list of documents
document_match_df=all_docs.iloc[doc_list]

#TODO: this may be fine, but feels like a fragile indexing method by using position
# potentially add some validations checks (length, spot-checks of specific obs, etc)

#save the new matched articles 
#document_match_df.to_csv('matched_new_articles.csv')

# =============================================================================
# Model Evaluation phase
# =============================================================================
validation_articles.set_index("PMID", inplace=True) #set index so that we can measure performance
#and see how many of the validation set are in our retrieved dataset

Matched_docs=document_match_df[document_match_df.index.isin(validation_articles.index)]

print("# of matched docs is {}".format(len(Matched_docs)))

# =============================================================================
# Now map to the new MeSH Terms of the retrieved documents
# =============================================================================
new_MeSH_list=document_match_df.MeSH_Descriptor.tolist()
new_MeSH_list1=[item for sublist in new_MeSH_list for item in sublist] #flatten list of lists 
pmid_df=all_docs[all_docs.index.isin(pmid_list)] # Get the row number associated with each term 

#we will use the original MeSH terms to find the most similar new ones
pmid_og_MeSH_list=pmid_df.MeSH_Descriptor.tolist()
pmid_og_MeSH_list=[item for sublist in pmid_og_MeSH_list for item in sublist ]
pmid_og_MeSH_list=[el for el in pmid_og_MeSH_list if el ] # drop empty ones
pmid_og_MeSH_list=list(set(pmid_og_MeSH_list))#make into a set 

# =============================================================================
# Modeling MeSH terms with word embeddings 
# =============================================================================
#Given the large number of MeSH terms returned from the queries, limit the number by modeling on TF-IDF and word embeddings

MeSH_kdt = KDTree(word_vectors, leaf_size=30, metric='euclidean')
#TODO: another instance here where we can try to hard set the number to return
MeSH_dist, MeSH_idx = MeSH_kdt.query(word_vectors,k=50)


#get the df_w2v index for the original MeSH terms so we can retrieve the 
# row number? or the word itself is the index?
# og_MeSH_terms_index=df_w2v.index.isin(pmid_og_MeSH_list)
MeSH_og_row_index=[]
for i,el in enumerate(df_w2v.index):
    if el in pmid_og_MeSH_list:
        MeSH_og_row_index.append(i)
#TODO: another fragile instance where this indexing method (by row position)
#makes the code inflexible and feels like it is easily breakable

# use df_w2v for indexing
# now get the words associated with each MeSH term
MeSH_distance_list=[]
MeSH_neighbor_list=[]
for i,el in enumerate(MeSH_idx):
    if MeSH_idx[i][0] in MeSH_og_row_index:
        MeSH_neighbor_list.append(MeSH_idx[:][i])
        MeSH_distance_list.append(MeSH_dist[:][i])

#get the most similar for each term 
MeSH_neighbor_info_df=pd.DataFrame({
    "Doc Index":MeSH_og_row_index,
    "Neighbor":MeSH_neighbor_list ,
    "Distance":MeSH_distance_list
})

#save
#MeSH_neighbor_info_df.to_csv('MeSH_w2v_neighbor_distance.csv')

# =============================================================================
# Now Get the new MeSH terms -- via word embeddings 
# =============================================================================
new_MeSH_list=MeSH_neighbor_info_df.Neighbor.tolist()
new_MeSH_list=[item for sublist in new_MeSH_list for item in sublist] #flattens the list of lists to a single list
new_MeSH_list=list(set(new_MeSH_list)) #count everything just once 

#returns the embeddings for the newest words 
new_MeSH_match=df_w2v.iloc[new_MeSH_list]
#TODO: there has to be a better way to do this index matching, too many steps
#possibly just have in one dataset or something, or find a more straightforward process

#get the new MeSH words as a list
w2v_MeSH_match_list=new_MeSH_match.index.tolist()
w2v_MeSH_match_list=list(set(w2v_MeSH_match_list))
w2v_MeSH_match_list=[item for item in w2v_MeSH_match_list if item]

# =============================================================================
# Use the new+larger MeSH list, compare to the original MeSH list, and run levenshtein distance measures 
# The result should form a cartesian product but shouldnt be a problem for speed-runtime
# more info: #https://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm
# =============================================================================

# read in the MeSH mapping file
MeSH_map=pd.read_table("MeSH_map.txt", sep=";", header=None)
MeSH_map.columns=['MeSH_desc','MeSH_str']
#TODO: Consider adding a Curling-scraping function to go and grab the newest 
#file directly from the MeSH FTP archive
# MeSH FTP archive: ftp://nlmpubs.nlm.nih.gov/online/mesh/

# get MeSH string(Unique-ID) for our original list 
og_MeSH_list_str=[]
for i, el in enumerate(MeSH_map['MeSH_desc']):
    if el in pmid_og_MeSH_list:
        og_MeSH_list_str.append(MeSH_map.iloc[i]['MeSH_str'])

import editdistance #for Lev Distance
#http://www.nltk.org/howto/metrics.html
# output 3 lists - MeSH IDs in our original docs , MeSH IDs in our new docs, Lev Distance value 
get_ipython().magic('timeit')
og_MeSH_str_list=[]
target_MeSH_str_list=[]
levenshtein_distance_list=[]
for el in MeSH_map['MeSH_str']:
    splt_str=el.split(".")
    if el not in og_MeSH_list_str and len(splt_str)>2: #only look at new MeSH terms and at least slightly specific
        for og_el in og_MeSH_list_str:
            #compute Levenshtein distance 
            dist=editdistance.eval(el,og_el)
            og_MeSH_str_list.append(og_el)
            target_MeSH_str_list.append(el)
            levenshtein_distance_list.append(dist)

Lev_df=pd.DataFrame({
    "og_MeSH_str": og_MeSH_str_list,
    "target_MeSH_str": target_MeSH_str_list,
    "Lev_Dist": levenshtein_distance_list
})

#create MeSH dictionary - for string ID and the value (description)
MeSH_dict=dict(zip(MeSH_map['MeSH_str'],MeSH_map['MeSH_desc']))

#pull back 
Lev_df['og_MeSH_desc']=Lev_df['og_MeSH_str'].map(MeSH_dict)

# =============================================================================
# Note: MeSH descriptions (e.g. Gout) have a one-to-many relationship with MeSH strings 
# e.g. these are all MeSH IDs for Gout: C05.550.354.500, C05.799.414, C16.320.565.798.368, C18.452.648.798.368
# =============================================================================
#add MesH value to lev dist df 
Lev_df['target_MeSH_desc']=Lev_df['target_MeSH_str'].map(MeSH_dict)

#old plotting code 
#from scipy import stats, integrate
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#import seaborn as sns
#sns.set(color_codes=True)
#sns.distplot(Lev_df['Lev_Dist'])

# get density - distribution of levenshtein distance
#get distribution of og MeSH terms by term-depth level 
#MeSH_map['split_count']=MeSH_map['MeSH_str'].apply(lambda row: len(row.split("."))) #get the nb of levels for every term
#TODO: not sure how useful this piece will be to include, consider removing or use somewhere else 

#subset entire MeSHmap DF for only original MeSHs 
og_MeSH_df=MeSH_map[MeSH_map['MeSH_desc'].isin(pmid_og_MeSH_list)]

# =============================================================================
# Not sure we want to use this next set of code in production
# Used to make graphics for EDA purposes 
# =============================================================================
#plot distro of number of string splits- measures how deep within the 
#MeSH hierarchy the term is 
#sns.distplot(og_MeSH_df['split_count'], bins=20, kde=False, rug=True)
#mas_5_levels=og_MeSH_df[og_MeSH_df['split_count']>6]
#MeSH_deep_terms=mas_5_levels['MeSH_desc'].tolist()
#MeSH_deep_terms=list(set(MeSH_deep_terms))
#import joypy

#get levenshtein distance for select descriptions
#joyplt_df=Lev_df[Lev_df['og_MeSH_desc'].isin(MeSH_deep_terms)]

# Joy Plot of MeSH terms
#fig, axes = joypy.joyplot(joyplt_df, by="og_MeSH_desc", column="Lev_Dist")
# # Reduce the possible number of matches based on Levenshtein distance

# =============================================================================
# Subsetting of MeSH terms based on Levenshtein
# =============================================================================
# keep if edit distance is less than 6 in relation to the original list
new_MeSH_keepers_Lev=Lev_df[(Lev_df['Lev_Dist']<6) & (Lev_df['og_MeSH_desc'].isin(og_MeSH_df['MeSH_desc']))]
#TODO: somewhat arbitrary number chosen for edit distance (represents roughly and edit of two whole levels in MeSH tree)
# consider making more data driven based upon the underlying data and distribution 
new_MeSH_keepers_Lev_list=new_MeSH_keepers_Lev['target_MeSH_desc'].tolist()
new_MeSH_keepers_Lev_list=list(set(new_MeSH_keepers_Lev_list))

# how many overlap with the word2vec model
final_MeSH_term_list=list(set(w2v_MeSH_match_list)&set(new_MeSH_keepers_Lev_list))

#import csv
#with open("Gout_MeSH_new_Query_terms.csv", 'w') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(final_MeSH_term_list)

#TODO: develop an inverse rank based simply upon how uncommon the terms are across the entire corpus of 10,000
# # cluster the new terms with aglomerative clustering

# use word embeddings for only these MeSH terms
new_MeSH_w2v_features=df_w2v[df_w2v.index.isin(final_MeSH_term_list)]
# =============================================================================
# Not sure if we want to include the clustering code in the final product
# =============================================================================
#from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage
#l = linkage(new_MeSH_w2v_features, method='complete', metric='seuclidean')

# calculate full dendrogram
#plt.figure(figsize=(25, 20))
#plt.title('Hierarchical Clustering Dendrogram')
#plt.ylabel('word')
#plt.xlabel('distance')

#dendrogram(
#    l,
#    truncate_mode='lastp',  # show only the last p merged clusters
#    p=25,  # show only the last p merged clusters
#    leaf_rotation=90.,  # rotates the x axis labels
#    leaf_font_size=12.,  # font size for the x axis labels
##     orientation='left',
##     leaf_label_func=lambda v: str(new_MeSH_w2v_features.index[v]) #dont use with truncation?
#)
#plt.show()

# Get the Clusters
#from scipy.cluster.hierarchy import fcluster
#clusters = fcluster(l,20, criterion='maxclust')

#unique, counts = np.unique(clusters, return_counts=True)
#print (np.asarray((unique, counts)).T)

#new_MeSH_w2v_features.to_csv("w2v_MeSH_clusters.csv")

# =============================================================================
# T-SNE plot
# =============================================================================
#new_MeSH_w2v_features.drop(['cluster_number'], axis=1, inplace=True)
#from sklearn.manifold import TSNE
#tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
#new_values = tsne_model.fit_transform(new_MeSH_w2v_features)

#use word embedding features 
#x = []
#y = []
#for value in new_values:
#    x.append(value[0])
#    y.append(value[1])
#labels=new_MeSH_w2v_features.index

#plt.figure(figsize=(16, 16)) 
#for i in range(len(x)):
#    plt.scatter(x[i],y[i])
#    plt.annotate(labels[i],
#                 xy=(x[i], y[i]),
#                 xytext=(5, 2),
#                 textcoords='offset points',
#                 ha='right',
#                 va='bottom')
#plt.show()
#

#  restrict to the 100 most common MeSH terms within our returned corpus 
# return freq counts for our 300+ terms within the 204 new articles
returned_MeSH_vocab=Matched_docs['MeSH_Descriptor'].tolist()
returned_MeSH_vocab=[el for sublist in  returned_MeSH_vocab for el in sublist ] #flatten
returned_MeSH_vocab=[el for el in returned_MeSH_vocab if el] # drop empties

# subset to only words in final MeSH list
returned_MeSH_vocab_final_subset=[el for el in returned_MeSH_vocab if el in new_MeSH_w2v_features.index ]

# get the most common terms for better T-SNE visualization AND 
#to sort and prioritize final MeSH results based on count
from collections import Counter
w_counts=Counter(returned_MeSH_vocab_final_subset)

w_counts_most_common=w_counts.most_common(100)
#save top MeSH results
with open("MeSH_term_count_of_returned.txt", encoding='utf-8-sig', mode='w') as f:
    for k,v in  w_counts.items():
        f.write( "{} {}\n".format(k,v) )

w_counts_most_common_list=[]
for x,_ in w_counts_most_common:
    w_counts_most_common_list.append(x)

# now subset our word 2vec list
subsetted_w2v_MeSH_list=new_MeSH_w2v_features[new_MeSH_w2v_features.index.isin(w_counts_most_common_list)]

#output final MeSH results 
subsetted_w2v_MeSH_list.to_csv("w2v_MeSH_top_n_terms_w_features.csv")

from sklearn.manifold import TSNE
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(subsetted_w2v_MeSH_list)

import matplotlib.pyplot as plt
%matplotlib inline
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
labels=subsetted_w2v_MeSH_list.index

plt.figure(figsize=(22, 20)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()
