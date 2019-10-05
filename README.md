# TEAM-B.-TASK-5
The Task is to "find movie similarities based on their plots summaries"

To all movie enthusiast out there the aim of this model is to be able to recommend a movie with similar plots to the one you have in mind. obviously we all have a particular genre of movie we would like to watch be it sci-fi, horror, romance and the likes
And more often than not there is usally a common factor so to say in these similar movies which can be seen in their plots.

People who are just interested in just the movies can skip this part. But you are welcome to learn

THE TECHNICAL PART

Something called Tokenization was used; so in layman's terms it is basically the splitting of a larger body of text into smaller lines
to do this NLTK is used which stands for "Natural Language Toolkit" 
which is a platform used for building Python programs that work with human language data for applying in statistical natural language processing (NLP). 
It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.

Stemming; this is also done with NLTK . It simply is word normalization technique used in the field of Natural Language Processing that are used to prepare text, words, and documents for further processing.
it generates root form of words. example words like ( waits, waited and waiting ) will have a basic root form of WAIT.

BRINGING TOKENIZE AND STEM TOGETHER

tokenize_and_stem: tokenizes (splits the synopsis into a list of its respective words (or tokens) and also stems each token which brings all the words are in their root form, which will lead to a better establishment of meaning as some of the non-root forms may not be present in the NLTK training corpus.

CREATION OF A TF-IDF VECTORIZER

Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining
The importance increases proportionally to the number of times a word appears in the document.

TF: Term Frequency, which measures how frequently a term occurs in a document. 
Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones.

TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. 
Thus we need to weigh down the frequent terms while scale up the rare ones.

So combining TF and IDF, gives TF-IDF

K MEANS CLUSTEREING

K-Means Clustering is an unsupervised machine learning algorithm. In contrast to traditional supervised machine learning algorithms, K-Means attempts to classify data without having first been trained with labeled data.
K-means is an algorithm which helps us to implement clustering in Python. The name derives from its method of implementation: the given sample is divided into K clusters where each cluster is denoted by the mean of all the items lying in that cluster.

DENOGRAM

Basically a dendrogram or tree diagram allows to illustrate the hierarchical organisation of several entities. Nothing much just a fancy name is just like your regular family tree in term of the structure.

Everthing above has been combined into creating a model which finds similar movie based on their plots summaries

The explanations above and the definition of terms is just to ease understanding of anyone reading the code.
