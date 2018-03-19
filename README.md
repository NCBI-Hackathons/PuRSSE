# SystematicReviews
Automated Systematic Reviews
NCBI March 2018


## Discovering The Smallest Precise Search Strategy for Pubmed from a set of Known PMIDs using Topic Modeling and Article Metadata

This project aims to create a pipeline for taking a set of known PMIDs and discovering the Shortest Precise Search Strategy (SPSS) for from Pubmed that (a) retrieves all the original PMIDs, and (b) retrieves other articles related to the original topic of the known PMIDs. This will use the article-level metadata provided by NLM and topic modeling, specifically LSA, which will then be used to create searches made up of MeSH headings and keywords. 

Known PMIDs (paste into interface) --> retrieve metadata --> LDA --> MAGIC --> Search Strategy (cut & paste into pubmed.gov)

We will also be creating a simple interface for testing the precision and recall of a search (PMIDs) against a known set of items (PMIDs). 


## Process

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



