# PuRSSE (Pubmed Research Search String Extraction)

<img src="https://raw.githubusercontent.com/NCBI-Hackathons/SystematicReviews/master/PuRSSE.png" height=30 /> This project aims to create a pipeline for taking a set of known PMIDs and discovering the Shortest Precise Search Strategy (SPSS) for Pubmed that (a) retrieves all the original PMIDs, and (b) retrieves other articles related to the original topic(s) extracted from the known PMIDs. This process uses the article-level metadata provided by NLM (title and abstract plus MeSH terms and keywords). Topic modeling, specifically TF-IDF, word embedding and latent Dirichlet allocation (LDA), are used on the title and abstract. TF-IDF and word embedding are used on the MeSH terms. The topics generated will be used to create search strings constructed from the corresponding MeSH headings and keywords. 

# Team Members

Melanie Huston, James Lavinder, Richard Lusk, Franklin Sayre

A prototype of the projected process is at [location]:<br>
Known PMIDs (paste list or search into interface) --> retrieve articles' metadata --> topic modeling and clustering --> search string construction using MeSH --> New Search Strategy (cut & paste search into PubMed.gov)

# Three Goals/Projects
1. Create clusters of articles based on topic modeling (TF-IDF, LDA, Word Embedding) from any PubMed-compliant XML file
2. Based on a set of known articles, find other articles that are similar using topic modeling (via either direct similarity comparison OR by building a new search string from metadata associated with topic clusters)
3. Compare known set of PMIDs with larger set of PMIDs to verify 100% recall of known set and ideal larger set size for optimal precision

# Why is this useful?
1. It's cool!
2. Researchers need ways of doing topic modeling on PubMed literature easily 
3. Creating a "shortest precise search strategy" based on a set of known PMIDs that retrieves those PMIDs and others like them could be useful for systematic reviews and other information retrieval tasks
4. Researchers/instructors need ways of quickly getting precision and recall scores for a set of PMIDs within another set of PMIDs

# How could this be used for systematic reviews
The first stage of creating a systematic review often involves taking a known set of articles (mostly available in PubMed and with PMIDs) and then iteratively looking through metadata and keywords to create an extensive search string that can find both those target articles and other similar articles, without retrieving too much. This could potentially be used to help with that process by recommending a search string. 

# How could this be used for topic modelling PubMed literature
This could be used to help with topic modeling PubMed literature by providing a pipeline that takes a PubMed compliant XML file (generated from PubMed.gov, or downloaded from PubMed FTP servers, or retrieved through EDirect) and outputing a set of topic models. This could be attached to other projects. 

# To do/Issues
- document ways of getting PubMed compliant XML files (ftp, PubMed.gov)
- see if EDirect gives compliant XML
- determine optimum way(s) to model topics (metadata and methods)
- create front end interface for end users
- optimum method for stemming medical terms

# Process

Get PubMed Data
- Download XML from NLM FTP. Approx 200GB. Benefits: all the data all the time. Negitives: with addition of new publications the list becomes out of date quickly, requires a lot of memory
- API. Slow. Doesn't require server space. Can't get everything. 
- Pubrunner
- EDirect Local Data Cache

Extract useful metadata from XML
- Metadata was extracted from PubMed XML files using python's lmxl model. The following features were extracted:
  - PMIDs
  - Title
  - Abstract Text
  - MeSH Major Headings
  - MeSH Subheadings
  - Keywords
- The title and abstract are concatanted together, this is for older PubMed records with no abstract, and then are cleaned for   processing. The cleaning process includes tokenizing, lowercasing, stemming and the removal of stop words. 
** Note: full text was not included in this extraction. **

# NLP Modeling on Title and Abstract
- TF-IDF is run on both Abstract+Title
- Document embeddings (Doc2Vec) are run on documents for features to measure similarity across documents. 
- Word embeddings (Word2vec) are run across the entire corpus to measure similarity and capture context across words. 
- Latent Dirichlet Allocation (LDA) is run for Topic Modeling purposes on Title and Abstract

# NLP Modeling on MeSH terms
- TF-IDF across MeSH terms to find co-occurence  
- Word embeddings (Word2vec) are run across the entire corpus of words

# Retrieve most similar documents to initial ones provided
- Determining document similarity through NLP feature vectors
- apply a nearest-neighbors approach (K-D Tree or K-D Ball) to retrieve the most similar documents to initial list of PMIDs
- Retrieve documents using pre-determined number PMIDs (e.g. n=2000) using a metric (e.g. cosine similarity)

# Map MeSH & Keyword Strings associated with newest retrieved documents
- Run hierarchical (agglomerative) clustering to find the different groupings of MeSH terms


Create Shortest Precise Search String 
- Penalize longer search strings - apply higher weight to MeSH terms deeper in MeSH tree
- Magic step 2
- Magic step 3
- Magic step 4

Test against known PMIDs
- Magic step 1
- Magic step 2
- Magic step 3
- Magic step 4



