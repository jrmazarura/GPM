# [GPyM_TM](https://github.com/jrmazarura/GPM)

**GPyM_TM** is a Python package to perform topic modelling, either through the use of a Dirichlet multinomial mixture model, or a Poisson model. Each of the above models is available within the package in a separate class, namely GSDMM utilizes the Dirichlet multinomial mixture model, while GPM makes use of the Poisson model to perform the text clustering respectively.  

## Preamble  
The aim of topic modelling is to extract latent topics from large corpora. GSDMM [1] assumes each document belongs to a single topic, which is a suitable assumption for some short texts. Given an initial number of topics, K, this algorithm clusters documents and extracts the topical structures present within the corpus. If K is set to a high value, then the model will also automatically learn the number of clusters.

[1]	Yin, J. and Wang, J., 2014, August. A Dirichlet multinomial mixture model-based approach for short text clustering. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 233-242).

## Getting Started:

The package is available [online](https://pypi.org/project/GPyM-TM/) for use within Python 3 enviroments.

The installation can be performed through the use of a standard 'pip' install command, as provided below: 

`pip install GPyM-TM`

## Prerequisites:

The package has several dependencies, namely: 

* numpy
* random
* math
* pandas
* re
* nltk
* gensim
