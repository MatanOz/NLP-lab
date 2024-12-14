# %%
import pandas as pd
from collections import Counter

# Load spam.csv
file_path = "spam.csv"  # Replace with the actual file path
data = pd.read_csv(file_path, encoding='latin-1')

# Clean up the dataset (adjust columns as per the dataset structure)
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[["label", "message"]]

# Compute statistics
num_messages = len(data)
num_spam = len(data[data["label"] == "spam"])
num_ham = len(data[data["label"] == "ham"])

# Word count statistics
word_counts = data["message"].str.split().map(len)
total_words = word_counts.sum()
avg_words = word_counts.mean()

# Most frequent words
all_words = " ".join(data["message"]).split()
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(5)

# Rare words (appear only once)
rare_words = [word for word, count in word_freq.items() if count == 1]
num_rare_words = len(rare_words)

# Output
print(f"Total messages: {num_messages}")
print(f"Spam messages: {num_spam}")
print(f"Non-spam (ham) messages: {num_ham}")
print(f"Total words: {total_words}")
print(f"Average words per message: {avg_words:.2f}")
print(f"5 most frequent words: {most_common_words}")
print(f"Number of rare words: {num_rare_words}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Spam vs. Ham count
spam_ham_counts = data["label"].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=spam_ham_counts.index, y=spam_ham_counts.values, palette="viridis")
plt.title("Spam vs. Ham Message Count", fontsize=16)
plt.xlabel("Message Type", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

# Plot Word Count Distribution
plt.figure(figsize=(10, 6))
sns.histplot(word_counts, kde=True, bins=30, color="blue")
plt.title("Word Count Distribution in Messages", fontsize=16)
plt.xlabel("Number of Words", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.show()
# %% 
import nltk
import spacy

# Download NLTK tokenizer resources
#nltk.download('punkt')

# NLTK Tokenization
nltk_tokens = data["message"].apply(nltk.word_tokenize)

# spaCy Tokenization
nlp = spacy.load("en_core_web_sm")
spacy_tokens = data["message"].apply(lambda x: [token.text for token in nlp(x)])

# Compare results
print(f"NLTK Tokenization example: {nltk_tokens.iloc[:10]}")
print(f"spaCy Tokenization example: {spacy_tokens.iloc[:10]}")

# %% Q5
# The results of tokenization using NLTK and spaCy are largely similar, as both tools effectively split the text into individual words, punctuation, and special characters. However, spaCy demonstrates greater linguistic context awareness, as seen in its handling of possessives and quotes (e.g., splitting 'Melle into ', and Melle), whereas NLTK treats such cases as single tokens. While NLTK is rule-based and simpler, spaCy's pre-trained models make it more reliable for tasks requiring grammatical accuracy. For basic tokenization, both are suitable, but spaCy is better for nuanced linguistic scenarios.
# %%
# Q6
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK lemmatizer resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# NLTK Lemmatization
lemmatizer = WordNetLemmatizer()
nltk_lemmatized = data["message"].apply(lambda x: [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)])

# spaCy Lemmatization
spacy_lemmatized = data["message"].apply(lambda x: [token.lemma_ for token in nlp(x)])

# Display example results
print(f"NLTK Lemmatization example: {nltk_lemmatized.iloc[0]}")
print(f"spaCy Lemmatization example: {spacy_lemmatized.iloc[0]}")



# Comparison of NLTK and spaCy Lemmatization
# The results of lemmatization with NLTK and spaCy show some key differences:

# Case Sensitivity:

# NLTK preserves the original case of the words ('Go', 'Available'), whereas spaCy converts words to lowercase ('go', 'available'), making its output more normalized.
# Verb Lemmatization:

# spaCy lemmatizes verbs more effectively. For example, 'got' becomes 'get' in spaCy, reflecting its dictionary base form. NLTK leaves 'got' unchanged.
# Punctuation:

# Both tools preserve punctuation like commas and ellipses (',', '..', '...'), treating them as tokens.
# Stopwords:

# Both retain common stopwords like 'in', 'only', and 'there'.

# %%
#q7

from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK resources
nltk.download('punkt')

# Initialize stemmers
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english")

# Apply all stemmers to the dataset
def apply_stemming_methods(text):
    tokens = word_tokenize(text)
    porter_stemmed = [porter_stemmer.stem(word) for word in tokens]
    lancaster_stemmed = [lancaster_stemmer.stem(word) for word in tokens]
    snowball_stemmed = [snowball_stemmer.stem(word) for word in tokens]
    return {
        "original": tokens,
        "porter": porter_stemmed,
        "lancaster": lancaster_stemmed,
        "snowball": snowball_stemmed,
    }

# Test stemming on the first message
example_message = data["message"].iloc[0]
stemmed_results = apply_stemming_methods(example_message)

# Display results
print("Original Tokens:", stemmed_results["original"])
print("Porter Stemmer:", stemmed_results["porter"])
print("Lancaster Stemmer:", stemmed_results["lancaster"])
print("Snowball Stemmer:", stemmed_results["snowball"])

"""
The results of the stemming methods show clear differences in their approaches. The Porter Stemmer provides a balanced and reliable general-purpose solution, effectively reducing words like 'crazy' to 'crazi' and 'available' to 'avail'. The Lancaster Stemmer, on the other hand, is more aggressive, over-simplifying words such as 'great' to 'gre' and 'amore' to 'am', which can lead to a loss of semantic clarity. The Snowball Stemmer produces results similar to the Porter Stemmer but offers more consistency and is better suited for multi-language tasks.

Since spaCy does not natively support stemming, we compared different stemming methods provided by NLTK instead. While the Lancaster Stemmer is overly aggressive, the Porter and Snowball Stemmers strike a balance between reducing words to their root forms and preserving interpretability, making them more suitable for most NLP applications.
"""


# %%
#q8
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize tools
nltk_lemmatizer = WordNetLemmatizer()
spacy_nlp = spacy.load("en_core_web_sm")

# Function to tokenize messages
def tokenize_messages(messages, method="NLTK"):
    if method == "NLTK":
        return messages.apply(word_tokenize)
    elif method == "spaCy":
        return messages.apply(lambda x: [token.text for token in spacy_nlp(x)])
    else:
        raise ValueError("Invalid method. Choose 'NLTK' or 'spaCy'.")

# Function to lemmatize tokens
def lemmatize_tokens(token_vectors, method="NLTK"):
    if method == "NLTK":
        return token_vectors.apply(lambda tokens: [nltk_lemmatizer.lemmatize(token) for token in tokens])
    elif method == "spaCy":
        return token_vectors.apply(lambda tokens: [spacy_nlp(token)[0].lemma_ for token in tokens])
    else:
        raise ValueError("Invalid method. Choose 'NLTK' or 'spaCy'.")

# Generate combined vector of all tokens
def generate_combined_vector(messages, method="NLTK"):
    token_vectors = tokenize_messages(messages, method)
    return [token for message_tokens in token_vectors for token in message_tokens]

# Find the first message that does not change the lemmatized vector
def find_first_removal_message(messages, method="NLTK"):
    combined_vector = generate_combined_vector(messages, method)
    combined_lemmatized = set(lemmatize_tokens(pd.Series([combined_vector]), method)[0])
    
    for index, message in messages.items():
        temp_messages = messages.drop(index)
        temp_vector = generate_combined_vector(temp_messages, method)
        temp_lemmatized = set(lemmatize_tokens(pd.Series([temp_vector]), method)[0])
        
        if temp_lemmatized == combined_lemmatized:
            # Proof: Return relevant details
            return {
                "index": index,
                "message": message,
                "original_lemmatized": combined_lemmatized,
                "modified_lemmatized": temp_lemmatized,
                "proof": "The lemmatized vectors are identical, so removing this message does not affect lemmatized tokens."
            }
    return None

# Example usage
messages = data["message"]  # Replace with your dataset column
method = "NLTK"  # Change to "spaCy" if preferred

result = find_first_removal_message(messages, method)
if result:
    print(f"for {method}:")
    print(f"Message index: {result['index']}")
    print(f"Message: {result['message']}")
    print(f"Original lemmatized vector: {result['original_lemmatized']}")
    print(f"Modified lemmatized vector: {result['modified_lemmatized']}")
    print(f"Proof: {result['proof']}")
else:
    print("No such message found.")

# %%
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize tools
nltk_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
spacy_nlp = spacy.load("en_core_web_sm")

# Tokenization Function
def tokenize_messages(messages, method="NLTK"):
    if method == "NLTK":
        return messages.apply(word_tokenize)
    elif method == "spaCy":
        return messages.apply(lambda x: [token.text for token in spacy_nlp(x)])
    else:
        raise ValueError("Invalid method. Choose 'NLTK' or 'spaCy'.")

# Lemmatization Function
def lemmatize_tokens(token_vectors, method="NLTK"):
    if method == "NLTK":
        return token_vectors.apply(lambda tokens: [nltk_lemmatizer.lemmatize(token) for token in tokens])
    elif method == "spaCy":
        return token_vectors.apply(lambda tokens: [spacy_nlp(token)[0].lemma_ for token in tokens])
    else:
        raise ValueError("Invalid method. Choose 'NLTK' or 'spaCy'.")

# Stemming Function
def stem_tokens(token_vectors, method="NLTK"):
    if method == "NLTK":
        return token_vectors.apply(lambda tokens: [porter_stemmer.stem(token) for token in tokens])
    else:
        raise ValueError("Stemming is only supported for 'NLTK'.")

# Generate Combined Vector
def generate_combined_vector(messages, method="NLTK"):
    token_vectors = tokenize_messages(messages, method)
    return [token for message_tokens in token_vectors for token in message_tokens]

# Unified Function to Find First Message
def find_first_removal_message(messages, method="NLTK", process="lemmatize"):
    print(f"Processing with method: '{method}' and process: '{process}'")
    combined_vector = generate_combined_vector(messages, method)
    
    if process == "lemmatize":
        combined_processed = set(lemmatize_tokens(pd.Series([combined_vector]), method)[0])
    elif process == "stem":
        if method == "spaCy":
            raise ValueError("Stemming is not supported with spaCy.")
        combined_processed = set(stem_tokens(pd.Series([combined_vector]), method)[0])
    else:
        raise ValueError("Invalid process. Choose 'lemmatize' or 'stem'.")
    
    for index, message in messages.items():
        temp_messages = messages.drop(index)
        temp_vector = generate_combined_vector(temp_messages, method)
        
        if process == "lemmatize":
            temp_processed = set(lemmatize_tokens(pd.Series([temp_vector]), method)[0])
        elif process == "stem":
            temp_processed = set(stem_tokens(pd.Series([temp_vector]), method)[0])
        
        if temp_processed == combined_processed:
            # Proof: Return relevant details
            return {
                "method": method,
                "process": process,
                "index": index,
                "message": message,
                "original_processed": combined_processed,
                "modified_processed": temp_processed,
                "proof": f"The {process}ed vectors are identical, so removing this message does not affect the results."
            }
    return None

# Q8 ##################
messages = data["message"]  # Replace with your dataset column

# Choose method ('NLTK' or 'spaCy') and process ('lemmatize' or 'stem')
method = "NLTK"  # Can also be "spaCy"
process = "lemmatize"  # Use "stem" for stemming

result = find_first_removal_message(messages, method, process)
if result:
    print(f"\nResults with method: '{result['method']}' and process: '{result['process']}':")
    print(f"Message index: {result['index']}")
    print(f"Message: {result['message']}")
    print(f"Original processed vector: {result['original_processed']}")
    print(f"Modified processed vector: {result['modified_processed']}")
    print(f"Proof: {result['proof']}")
else:
    print(f"\nNo such message found using method: '{method}' and process: '{process}'.")

# %%
#Q9#############
messages = data["message"]  # Replace with your dataset column

# Choose method ('NLTK' or 'spaCy') and process ('lemmatize' or 'stem')
method = "NLTK"  # Can also be "spaCy"
process = "stem"  # Use "stem" for stemming

result = find_first_removal_message(messages, method, process)
if result:
    print(f"\nResults with method: '{result['method']}' and process: '{result['process']}':")
    print(f"Message index: {result['index']}")
    print(f"Message: {result['message']}")
    print(f"Original processed vector: {result['original_processed']}")
    print(f"Modified processed vector: {result['modified_processed']}")
    print(f"Proof: {result['proof']}")
else:
    print(f"\nNo such message found using method: '{method}' and process: '{process}'.")

# %%
#Q10#############
messages = data["message"]  # Replace with your dataset column

# Choose method ('NLTK' or 'spaCy') and process ('lemmatize' or 'stem')
method = "NLTK"  # Can also be "spaCy"
process = "lemmatize"  # Use "stem" for stemming

result = find_first_removal_message(messages, method, process)
if result:
    print(f"\nResults with method: '{result['method']}' and process: '{result['process']}':")
    print(f"Message index: {result['index']}")
    print(f"Message: {result['message']}")
    print(f"Original processed vector: {result['original_processed']}")
    print(f"Modified processed vector: {result['modified_processed']}")
    print(f"Proof: {result['proof']}")
else:
    print(f"\nNo such message found using method: '{method}' and process: '{process}'.")



# Choose method ('NLTK' or 'spaCy') and process ('lemmatize' or 'stem')
method = "spaCy"  # Can also be "spaCy"
process = "lemmatize"  # Use "stem" for stemming

result = find_first_removal_message(messages, method, process)
if result:
    print(f"\nResults with method: '{result['method']}' and process: '{result['process']}':")
    print(f"Message index: {result['index']}")
    print(f"Message: {result['message']}")
    print(f"Original processed vector: {result['original_processed']}")
    print(f"Modified processed vector: {result['modified_processed']}")
    print(f"Proof: {result['proof']}")
else:
    print(f"\nNo such message found using method: '{method}' and process: '{process}'.")

# %%
