"""
cc4876 Kassey Chang
"""
#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

import string





def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # Retrieve all synsets for the provided lemma and pos
    synsets = wn.synsets(lemma, pos)
    
    # For each synset, retrieve all lemmas
    lemmas = [l.name() for s in synsets for l in s.lemmas()]
    
    # Convert the lemma names to their string representation and replace underscores with spaces
    candidates = set([l.replace('_', ' ') for l in lemmas])
    
    # Remove the input lemma from the set of candidates
    candidates.discard(lemma)
    
    # Return the set candidates
    return candidates
    

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2
    # Extract the target lemma and pos from the context
    lemma = context.lemma
    pos = context.pos
    
    # Call get_candidates() and get the candidate synonyms
    candidates = get_candidates(lemma, pos)
    
    # Stores the final scores for candidates
    candidate_scores = {}
    
    synsets = wn.synsets(lemma, pos)
    
    # Iterate all synsets
    for synset in synsets:
        # Calculate overlap score 'a' for synset (set it to a constant placeholder for now)
        a = 1
        
        # Calculate 'b': frequency for <s,t> where t is the target lemma
        target_lemma_freq = sum([l.count() for l in synset.lemmas() if l.name() == lemma])
        
        for lemma_syn in synset.lemmas():
            candidate = lemma_syn.name().replace("_", " ")
            if candidate in candidates:
                # Calculate 'c': frequency for <s,w> where w is the candidate synonym
                c = lemma_syn.count()
                
                # Calculate aggregate score: 1000*a + 100*b + c
                aggregate_score = 1000 * a + 100 * target_lemma_freq + c
                
                # Store the aggregate score for the candidate
                candidate_scores[candidate] = aggregate_score
    
    # Sort candidates by aggregate score and select the highest (excluding target lemma)
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top candidate if available
    for candidate, _ in sorted_candidates:
        if candidate != lemma:
            return candidate
    
    # Return None if no suitable candidate found
    return None



def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3
    lemma = context.lemma
    pos = context.pos
    
    # Construct the complete sentence from left_context and right_context
    sentence = " ".join(context.left_context + [context.word_form] + context.right_context)
    
    synsets = wn.synsets(lemma, pos)
    
    # Tokenize and normalize the context
    stop_words = set(stopwords.words('english'))
    context_tokens = set(tokenize(sentence)) - stop_words
    
    best_score = -1
    best_lemma = None
    
    for synset in synsets:
        # Calculate 'a' the overlap score
        definition = set(tokenize(synset.definition())) - stop_words
        examples = set(tokenize(' '.join(synset.examples()))) - stop_words
        
        hypernyms = set()
        for hypernym in synset.hypernyms():
            hypernyms.update(set(tokenize(hypernym.definition())) - stop_words)
            hypernyms.update(set(tokenize(' '.join(hypernym.examples()))) - stop_words)
        
        a = len(context_tokens.intersection(definition, examples, hypernyms))
        
        # Calculate 'b': frequency for <s,t> where t is the target lemma
        b = sum([l.count() for l in synset.lemmas() if l.name() == lemma])
        
        for lemma_syn in synset.lemmas():
            candidate = lemma_syn.name().replace("_", " ")
            
            # Calculate 'c': frequency for <s,w> where w is the candidate synonym
            c = lemma_syn.count()
            
            # Calculate aggregate score: 1000*a + 100*b + c
            aggregate_score = 1000 * a + 100 * b + c
            
            if aggregate_score > best_score and candidate != lemma:
                best_score = aggregate_score
                best_lemma = candidate
    
    return best_lemma


        
   
# Part 4
class Word2VecSubst(object):
   
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # Obtain a set of possible synonyms from WordNet
        candidates = get_candidates(context.lemma, context.pos)
        
        # Filter out candidates that are not in the Word2Vec vocabulary
        valid_candidates = [c for c in candidates if c in self.model]
        
        # If no valid candidates, return None
        if not valid_candidates:
            return None
        
        # Return the synonym that is most similar to the target word, according to the Word2Vec embeddings
        most_similar = max(valid_candidates, key=lambda candidate: self.model.similarity(context.lemma, candidate))
        
        return most_similar

# Part 5
class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Obtain a set of candidate synonyms.
        candidates = get_candidates(context.lemma, context.pos)
        
        # Convert the information in context into a suitable masked input representation
        # Construct the sentence with a [MASK] token at the exact position of the target lemma
        sentence = ['[CLS]'] + context.left_context + ['[MASK]'] + context.right_context + ['[SEP]']
        input_toks = self.tokenizer.encode(sentence, add_special_tokens=False)
        
        # Find the index of the [MASK] token
        mask_idx = input_toks.index(self.tokenizer.mask_token_id)
        
        # Convert to numpy array with an extra batch dimension
        input_mat = np.array(input_toks).reshape((1,-1))
        
        # Run the DistilBERT model on the input representation
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        
        # Select the highest-scoring word from the set of WordNet-derived candidate synonyms
        # Sort all words based on BERT's output
        best_words_ids = np.argsort(predictions[0][mask_idx])[::-1]
        
        # Iterate through the sorted BERT predictions to find the first word that matches a candidate
        for word_id in best_words_ids:
            word = self.tokenizer.decode([word_id])
            if word in candidates:
                return word

# Part 6 ---- Best performance in all predictors
class AdvancedBertPredictor(object):
    """
    AdvancedBertPredictor:
    - Derived from BertPredictor
    - New features:
        - Dual Input: Masked & unmasked sentences.
        - Combined Predictions: Adds predictions from both.
        - Flexible Selection: Increases contextual relevance.
    """

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)

        # Create two different input sequences: one with MASK and one with the original lemma
        input_with_mask = context.left_context + ['[MASK]'] + context.right_context
        input_with_lemma = context.left_context + [context.lemma] + context.right_context

        # Convert to token IDs
        input_ids_mask = self.tokenizer.convert_tokens_to_ids(input_with_mask)
        input_ids_lemma = self.tokenizer.convert_tokens_to_ids(input_with_lemma)

        # Predict using the model for both inputs
        predictions_mask = self.model.predict(np.array(input_ids_mask).reshape(1, -1), verbose=0)
        predictions_lemma = self.model.predict(np.array(input_ids_lemma).reshape(1, -1), verbose=0)

        # Merge the predictions by adding them
        combined_predictions = predictions_mask[0] + predictions_lemma[0]

        # Find the position of MASK or lemma in the input sequence
        position = len(context.left_context)

        # Sort tokens based on their score in combined_predictions
        best_tokens = np.argsort(combined_predictions[0][position])[::-1]


        best_words = self.tokenizer.convert_ids_to_tokens(best_tokens)

        # Return the first word that matches our candidates
        for word in best_words:
            if word in candidates:
                return word

        return None


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # Part 4
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # Part 5
    # predictor = BertPredictor()

    # Part 6
    predictor = AdvancedBertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 

        # Part 2
        # prediction = wn_frequency_predictor(context)
        """
        Total = 298, attempted = 298
        precision = 0.107, recall = 0.107
        Total with mode 206 attempted 206
        precision = 0.146, recall = 0.146
        """

        # Part 3
        # prediction = wn_simple_lesk_predictor(context)
        """
        Total = 298, attempted = 298
        precision = 0.112, recall = 0.112
        Total with mode 206 attempted 206
        precision = 0.146, recall = 0.146
        """

        # Part 4 
        # prediction = predictor.predict_nearest(context)
        """
        Total = 298, attempted = 298
        precision = 0.115, recall = 0.115
        Total with mode 206 attempted 206
        precision = 0.170, recall = 0.170
        """

        # Part 5
        # prediction = predictor.predict(context)
        """
        Total = 298, attempted = 298
        precision = 0.123, recall = 0.123
        Total with mode 206 attempted 206
        precision = 0.184, recall = 0.184
        """

        # Part 6
        prediction = predictor.predict(context)
        """
        Total = 298, attempted = 298
        precision = 0.145, recall = 0.145
        Total with mode 206 attempted 206
        precision = 0.228, recall = 0.228
        """

        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
