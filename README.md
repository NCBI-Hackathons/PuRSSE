# SystematicReviews
Automated Systematic Reviews
NCBI March 2018


## Discovering The Smallest Precise Search Strategy for Pubmed from a set of Known PMIDs using Topic Modeling and Article Metadata

This project aims to create a pipeline for taking a set of known PMIDs and discovering the Shortest Precise Search Strategy (SPSS) for from Pubmed that (a) retrieves all the original PMIDs, and (b) retrieves other articles related to the original topic of the known PMIDs. This will use the article-level metadata provided by NLM and topic modeling, specifically LSA, which will then be used to create searches made up of MeSH headings and keywords. 

Known PMIDs (paste into interface) --> retrieve metadata --> LDA --> MAGIC --> Search Strategy (cut & paste into pubmed.gov)

We will also be creating a simple interface for testing the precision and recall of a search (PMIDs) against a known set of items (PMIDs). 

