import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')  
neuralcoref.add_to_pipe(nlp)

# Sample text to use to do coreference resolution
texts = ["Harry starts with number0 apples . He buys number1 more . How many apples does Harry end with ?","Hello world"]

temp=[]
# Retrieve the spaCy Doc (composed of Tokens)
for sent in texts:
    doc = nlp(sent)  
    temp.append(doc._.coref_resolved)
# Retrieve a list of all the clusters of corefering mentions using the doc._.coref_clusters attribute 
# print(doc._.coref_clusters)
print(temp)
# Replace corefering mentions with the main mentions in each cluster by using the doc._.coref_resolved attribute.
# print(doc._.coref_resolved)
