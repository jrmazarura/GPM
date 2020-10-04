import numpy as np
import random
import math
import re, nltk
from gensim.corpora import Dictionary

class DMM:
    def __init__(self, corpus, nTopics, alpha = None, beta = None, nTopWords = None, iters = None):
        id2word = Dictionary(corpus)
        
        corpus = [id2word.doc2bow(doc) for doc in corpus]
        
        self.alpha = 0.1 if alpha is None else alpha
        self.beta = 0.1 if beta is None else beta
        self.nTopwords = 10 if nTopWords is None else nTopWords
        self.iters = 15 if iters is None else iters
        
        print("corpus=%d, words=%d, K=%d, a=%f, b=%f, nTopWords=%d, iters=%d" % (len(corpus), len(id2word), nTopics, self.alpha,self.beta, self.nTopwords, self.iters))
        self.nTopics = nTopics         # number of topics
        self.corpus = corpus
        self.id2word = id2word

        self.numDocuments = len(corpus)
        self.topicAssignments = []
        self.docTopicCount = [] #number of documents in topic (m_z)
        self.topicWordCount = [] #number of occurrences of word w in topic z (n_z_w)
        self.sumTopicWordCount = [] #number of words in topic z (n_z)
        self.betaSum=len(self.id2word)*self.beta
        self.conditional_prob = []
        self.finalAssignment = []
        
        
        self.psi = []
        self.theta = np.zeros((self.numDocuments, self.nTopics))
        
        self.output = ''
        self.twords = 10
        
    def topicAssigmentInitialise(self):
        self.docTopicCount = [0 for x in range(self.nTopics)] #initialise
        self.sumTopicWordCount = [0 for x in range(self.nTopics)] #initialise

        for i in range(self.nTopics):
            self.topicWordCount.append([0 for x in range(len(self.id2word))]) #initialise

        for d in range (self.numDocuments):
            topic = random.randint(0,self.nTopics-1) #for each document, sample a topic
            self.docTopicCount[topic]+=1 #update number of documents in topic (m_z)
            N_d = np.sum([word[1] for word in self.corpus[d]]) #number of words in document
            self.sumTopicWordCount[topic]+= N_d #update number of words in topic
 
            for j in range (len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]]+=word[1] #update number of occurences of word w in document

            self.topicAssignments.append(topic) #record the current topic of this document
        #print(self.docTopicCount)
        #print(self.sumTopicWordCount)
        #print(self.topicWordCount)
        #print('\n')
        
    def nextDiscrete(self,a):
        b = 0.

        for i in range(len(a)):
            b+=a[i]
        
        r = random.uniform(0.,1.)*b
		
        b=0.
        #print(r)
        for i in range (len(a)):
            b+=a[i]
            if(b>r):
                return i
        return len(a)-1
    
    def sampleInSingleIteration(self,x):
        print ("iteration: "+str(x))
        #print(self.topicAssignments)
        #print('\n')
        for d in range(self.numDocuments):
            #print ("document: "+str(d))
            topic = self.topicAssignments[d] #record the current cluster of d
            #print ("topic assignment: "+str(topic))
            #print ("topic assignment before: "+str(self.docTopicCount))
            self.docTopicCount[topic]-=1 #remove this document from assigned topic
            #print ("topic assignment after: "+str(self.docTopicCount))
            N_d = np.sum([word[1] for word in self.corpus[d]]) #number of words in document
            #print("number of words in doc " + str(N_d))
            #print ("topic words before: "+str(self.sumTopicWordCount))
            self.sumTopicWordCount[topic]-= N_d #remove number of words from this topic
            #print ("topic words  after: "+str(self.sumTopicWordCount))

            #print("nzw before" + str(self.topicWordCount))
            for j in range(len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]]-=word[1] #remove number of occurences of word w in document
            #print("nzw after " + str(self.topicWordCount))
            #sample a topic for d:
            for t in range(self.nTopics):
                self.conditional_prob[t] = (self.docTopicCount[t]+self.alpha)#/(self.numDocuments - 1 + self.ntopics*self.alpha)
                
                
                #print("document: " + str(self.corpus[d]))
                i = 0
                for w in range(len(self.corpus[d])):
                    
                    #print('w ' + str(w))
                    word = self.corpus[d][w] 
                    #print("word: " + str(word))                    
                    for j in range(word[1]):
                        #print('i ' + str(i))
                        #print('j ' + str(j))
                        i = i + 1
                
                        self.conditional_prob[t] *= (self.topicWordCount[t][word[0]]+self.beta + (j+1) - 1) /  \
                        (self.sumTopicWordCount[t] + self.betaSum + i - 1)
                
				
            #print(self.conditional_prob)
            topic = self.nextDiscrete(self.conditional_prob)
            self.theta[d,:] =  self.conditional_prob / np.sum(self.conditional_prob)

            self.docTopicCount[topic]+=1
            self.sumTopicWordCount[topic] += N_d #remove number of words from this topic

            for j in range(len(self.corpus[d])):
               word = self.corpus[d][j]
               self.topicWordCount[topic][word[0]] += word[1] #remove number of occurences of word w in document

            self.topicAssignments[d] = topic
        
        return len(np.unique(self.topicAssignments))


    def inference(self):
        out=[]
        self.conditional_prob = [0 for x in range(self.nTopics)]
        for x in range(self.iters):
            numtopics=self.sampleInSingleIteration(x)
            out.append(numtopics)
        return out  
        
        
    def worddist(self):
        """get topic-word distribution"""
        
        psi_file = open("%s_DMM_psi_Kstart.psi" % (self.nTopics),"w")
        self.psi = np.zeros((len(self.id2word), self.nTopics))
        for t in range(self.nTopics):
            norm = np.sum(self.topicWordCount[t])
            for w in range(len(self.id2word)):
                self.psi[w,t] = (self.topicWordCount[t][w] + self.beta)/(norm + self.betaSum)
                psi_file.write(str(self.psi[w,t]) + " ")    
            psi_file.write("\n")
        psi_file.close() 

        file = open("%s_DMM_topicAssignments_Kstart.txt" % (self.nTopics),"w")
		#for i in range(self.numDocuments):
        [file.write(str(self.topicAssignments[i])+"\n") for i in range(self.numDocuments)]
        file.close
        #print(self.topicAssignments)
        
        file2 = open("%s_DMM_selectedTopics_Kstart.txt" % (self.nTopics),"w")
        for selected_topic in np.unique(self.topicAssignments):
            file2.write(str(selected_topic)+"\n")
        file2.close
        
        self.finalAssignment = np.unique(self.topicAssignments)

        selected_psi = np.zeros([np.shape(self.psi)[0],len(self.finalAssignment)])
        i=0
        for kk in self.finalAssignment:
            selected_psi[:,i] = self.psi[:,kk]
            i=i+1               

        theta_file = open("%s_DMM_thetas_Kstart.theta" % (self.nTopics),"w")
        for m in range(self.numDocuments):
            for k in range(self.nTopics):
                theta_file.write(str(self.theta[m,k]) + " ")    
            theta_file.write("\n")
        theta_file.close()
        
        selected_theta = np.zeros([np.shape(self.theta)[0],len(self.finalAssignment)])
        j=0
        for jj in self.finalAssignment:
            selected_theta[:,j] = self.theta[:,jj]
            j=j+1
        
        return self.psi, self.theta, selected_psi, selected_theta

    def writeTopicAssignments(self): 
        file = open("%s_DMM_topicAssignments_Kstart.txt" % (self.nTopics),"w")
		#for i in range(self.numDocuments):
        [file.write(str(self.topicAssignments[i])+"\n") for i in range(self.numDocuments)]
        file.close
        #print(self.topicAssignments)
        
        file2 = open("%s_DMM_selectedTopics_Kstart.txt" % (self.nTopics),"w")
        for selected_topic in np.unique(self.topicAssignments):
            file2.write(str(selected_topic)+"\n")
        file2.close
        print(np.unique(self.topicAssignments))
        
        self.finalAssignment = np.unique(self.topicAssignments)
        
        return self.finalAssignment


    def writeTopTopicalWords(self, selected_topics):
        file = open("%s_DMM_topics_Kstart.topWords" % (self.nTopics),"w") 
        coherence_index_all=[]
        for t in selected_topics:
            wordCount = {w:self.topicWordCount[t][w] for w in range(len(self.id2word))}
			
            count =0
            string=""
            coherence_index_per_topic=[]
			
            for index in sorted(wordCount, key=wordCount.get, reverse=True):
                coherence_index_per_topic.append(index)
                string += self.id2word[index]+" "
                count+=1
                #print(count)
                if count>=self.twords:
                    file.write(string+"\n") 
                    print(string)
                    break
            coherence_index_all.append(coherence_index_per_topic)
        file.close()
        return(coherence_index_all)
        
    def coherence(self,topic_word,T):
        coherence_corpus=[]
        for a in self.corpus:
            coherence_doc=[]
            for b in a:
                coherence_doc.append(b[0])
            coherence_corpus.append(coherence_doc)
###
        nTopics = T 
        nTopWords = len(topic_word[0]) #number of top words per topic
        nDocs = len(coherence_corpus) #number of documents
        epsilon = 1 #smoothing parameter
        coherence = []     
        
                
        for t in range(0,nTopics): #calculate coherence
                       
            for vj in range(1,nTopWords): #each word in topic t
                  
                Dvj = 0
                for d in range(0,nDocs): 
                    if (topic_word[t][vj] in coherence_corpus[d]): #check how many docs contain word vj
                        Dvj += 1
                        
                for vi in range(0,vj): 
                    Dvjvi = 0
                    for d in range(0,nDocs): 
                        
                        if (topic_word[t][vj] in coherence_corpus[d]) and (topic_word[t][vi] in coherence_corpus[d]): #check how many docs contain both word vj and vi
                            Dvjvi += 1
                    
                    coherence.append(math.log((Dvjvi+epsilon)/float(Dvj),10))
                    
        print ("average topic: ", np.sum(coherence)/nTopics)

        return np.sum(coherence)/nTopics
 

def load_file(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        doc = re.findall(r'\w+(?:\'\w+)?',line)
        if len(doc)>0:
            corpus.append(doc)
    f.close()
    return corpus