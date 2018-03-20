# PuRSSE
# Pubmed Research Search String Extraction

<img src="https://raw.githubusercontent.com/NCBI-Hackathons/SystematicReviews/master/PuRSSE.png" />

This project aims to create a pipeline for taking a set of known PMIDs and discovering the Shortest Precise Search Strategy (SPSS) for from Pubmed that (a) retrieves all the original PMIDs, and (b) retrieves other articles related to the original topic of the known PMIDs. This will use the article-level metadata provided by NLM and topic modeling, specifically  latent Dirichlet allocation (LDA), which will then be used to create searches made up of MeSH headings and keywords. 

Known PMIDs (paste into interface) --> retrieve metadata --> LDA --> MAGIC --> Search Strategy (cut & paste into pubmed.gov)

# Three Goals/Projects
1. Create clusters of articles based on topic modeling (LDA, Word Embedding) from any Pubmed-compliant XML file
2. Based on a set of known articles, find other articles that are similar using topic modeling (via either direct similarity comparison OR by building a new search string from metadata associated with topic clusters)
3. Compare known set of PMIDs with larger set of PMIDs and find precision and recall

# Why is this useful?
1. It's cool!
2. Researchers need ways of doing topic modeling on pubmed literature easily 
3. Creating a "shortest precise search strategy" based on a set of known PMIDs that retrieves those PMIDs and others like them could be useful for systematic reviews and other information retrieval tasks
4. Researchers/instructors need ways of quickly getting precision and recall scores for a set of PMIDs within another set of PMIDs

# How could this be used for systematic reviews
The first stage of creating a systematic review often involves taking a known set of articles (mostly available in Pubmed and with PMIDs) and then iteratively looking through metadata and keywords to create an extensive search string that can find both those target articles and other similar articles, without retrieving too much. This could potentially be used to help with that process by recommending a search string. 

# How could this be used for topic modelling pubmed literature
This could be used to help with topic modeling pubmed literature by providing a pipeline that takes a pubmed compliant XML file (generated from pubmed.gov, or downloaded from pubmed FTP servers, or retrieved through EDirect) and outputing a set of topic models. This could be attached to other projects. 

# To do
- Document ways of getting pubmed compliant XML files (ftp, pubmed.gov)
- see if EDirect gives compliant XML
- 

# Process

Get Pubmed Data
- Download XML from NLM FTP. Approx 200GB. Benefits: all the data all the time. Negitives: out of date fast, requires a lot of space
- API. Slow. Doesn't require server space. Can't get everything. 
- Pubrunner
- EDirect Local Data Cache

Extract useful metadata from XML
- Abstract Text
- Title
- MeSH Major Headings
- MeSH Subheadings (?)
- Keywords
- PMIDs

Topic Modelling
- Magic step 1
- Magic step 2
- Magic step 3
- Magic step 4

Find MeSH & Keyword Strings associated with Topics
- Magic step 1
- Magic step 2
- Magic step 3
- Magic step 4

Create Shortest Precise Search String 
- Magic step 1
- Magic step 2
- Magic step 3
- Magic step 4

Test against known PMIDs
- Magic step 1
- Magic step 2
- Magic step 3
- Magic step 4



