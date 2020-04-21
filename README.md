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

# GSDMM

## Function and class description:

The class is named **GSDMM**, while the function itself is named **DMM**.

The function can take 6 possible arguments, two of which are required, and the remaining 4 being optional. 

### The required arguments are: 

* **corpus** - text file, which has been cleaned and loaded into Python. That is, the text should all be lowercase, all punctuation and numbers should have also been removed. 
* **nTopics** - the number of topics.

### The optional requirements are:

* **alpha**, **beta** - these are the distribution specific parameters.(**The defaults for both of these parameters are 0.1.**)
* **nTopWords** - number of top words per a topic.(**The default is 10.**)  
* **iters** - number of Gibbs sampler iterations.(**The default is 15.**)

## Output:

The function provides several components of output, namely:
* **psi** - topic x word matrix.
* **theta** - document x topic matrix.
* **topics** - the top words per topic. 
* **assignments** - the topic numbers of selected topics only, as well as the final topic assignments.
* **Final k** - the final number of selected topics.
* **coherence** - the coherence score, which is a performance measure.
* **selected_theta**
* **selected_psi**

# Example Usage:

A more comprehensive [tutorial]https://github.com/CAIR-ZA/GPyM_TM/blob/master/Tutorial.ipynb) is also available.

### Installation;

Run the following command within a Python command window:

`pip install GPym_TM`

### Implementation;

Import the package into the relevant python script, with the following: 

`from GSDMM import GSDMM`

> Call the class:

#### Possible examples of calling the function are as follows:

`data_dmm = GSDMM.DMM(corpus, nTopics)`

`data_dmm = GSDMM.DMM(corpus, nTopics, alpha = 0.5, beta = 0.5, nTopWords = 15, iters = 10)`

### Results;

The output obtained appears as follows: 

![Post](/Images/Post.png)

# GPM

The class to perform topic modelling through the use of a Poisson model is currently still under development, and will be added to the package at a later date with a updated version. 

## Built With:

[Google Collab](https://colab.research.google.com/notebooks/intro.ipynb) - Web framework

[Python](https://www.python.org/) - Programming language of choice

[Pypi](https://pypi.org/) - Distribution

## Authors:

[Jocelyn Mazarura](https://github.com/jrmazarura/GPM)


## Co-Authors:

[Alta de Waal](https://github.com/altadewaal)

[Ricardo Marques](https://github.com/RicSalgado)


## License:

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments:

University of Pretoria 
![Tuks Logo](/Images/UPlogohighres.jpg)
