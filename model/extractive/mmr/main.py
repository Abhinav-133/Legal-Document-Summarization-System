from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def generate_mmr_summary(text, desired_length_words=40000, lambda_param=0.6):
    def split_into_sentences(text):
        return sent_tokenize(text)
    
    def count_word(text):
        words = word_tokenize(text)
        words = [w for w in words if w not in string.punctuation]
        return len(words)
    
    def compute_pairwise_similarities(documents):
        vectorizer = TfidfVectorizer(smooth_idf=False)
        X = vectorizer.fit_transform(documents)
        X = X.toarray()
        rows = X.shape[0]
        pairwise_sim = []
        max_sim = []
        
        for i in range(rows):
            max_val = -1
            pairwise_sim.append([])
            for j in range(rows):
                if i != j:
                    sim = np.sum(np.multiply(X[i], X[j]))
                    pairwise_sim[-1].append(sim)
                    max_val = max(max_val, sim)
                else:
                    pairwise_sim[-1].append(1)  # similarity with itself
            max_sim.append(max_val)
        return pairwise_sim, max_sim
    
    def maximum_similarity(pair_sim, selected_indices, index):
        if not selected_indices:
            return 0
        return max(pair_sim[index][j] for j in selected_indices)

    # Split the text into sentences
    doc = split_into_sentences(text)
    if not doc:
        return ""  # No sentences found

    word_counts = [count_word(sent) for sent in doc]
    pair_similarity, l3 = compute_pairwise_similarities(doc)

    # Initialize variables for MMR process
    summary_indices = []
    summary_word_count = 0

    while summary_word_count < desired_length_words and len(summary_indices) < len(doc):
        best_score = -float('inf')
        best_index = -1
        for i in range(len(doc)):
            if i in summary_indices:
                continue
            max_sim = maximum_similarity(pair_similarity, summary_indices, i)
            mmr_score = lambda_param * l3[i] - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = i
        
        if best_index == -1:
            break

        summary_indices.append(best_index)
        summary_word_count += word_counts[best_index]
        
        # If the word count goes over, stop adding further
        if summary_word_count >= desired_length_words:
            break

    # Select the sentences based on the best indices
    selected_sentences = [doc[i] for i in summary_indices]
    summary = " ".join(selected_sentences)
    return summary

text = """N: Criminal Appeal No. 8 of 1951.
Appeal from an Order of the High Court of Bombay (Bavdekar and Chainani JJ.) dated 20th February, 1950, in Criminal Appeal No. 106 of 1950 arising out of an order dated 9th January, 1950, of the Presidency Magistrate, 19th Court, Esplanade, Bombay, in Case No. 10879/P of 1949.
The facts are stated in the judgment.
Iswarlal C. Dalai and R.B. Dalai, for the appellant.
C.K. Daphtary, Solicitor General for India (G. N. Joshi, with him) for the Republic of India (respondent No. 1).Jindra Lal for the respondent No. 2. 1952.
February 1.
The Judgment of the Court was deliv ered by CHANDRASEKHARA AIYAR J.
The facts out of which this Crimi nal Appeal has arisen are not long.
The appellant, W.H. King, who is carrying on a business in Bombay under the name and style of Associated Commercial Enterprises, was the tenant of a flat on the second floor of a building called "Ganga Vihar", Marine Drive, Bombay, which belongs to a lady named Durgeshwari Devi.
"""
summary = generate_mmr_summary(text)
print(summary)
