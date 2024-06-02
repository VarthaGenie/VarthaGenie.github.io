# Vartha Genie: Building a Telugu News Article Summarizer

## Team Members
- Raghu Prashanka
- Charan Sai
- Badreesh
- Dhanush Reddy
- Sai Varun

## Problem Statement
This project aims to develop a Natural Language Processing (NLP) model capable of summarizing Telugu news articles. The model will take a Telugu news article URL as input and output a concise and informative summary of the article content.

## Motivation
There is a growing demand for consuming news content in regional languages like Telugu. This project seeks to bridge this gap by creating a Telugu news article summarizer. This model will empower Telugu speakers to:
- Quickly grasp the key points of news articles without having to read the entire piece.
- Stay informed about current events in their native language.
- Enhance their overall engagement with Telugu news media.

## Technical Approach

### Data Collection
To build a robust Telugu news article summarizer, we took a large corpus of Telugu news articles. This data served as the foundation for training the NLP model. Here's a breakdown of the data collection process:
- **Integration with Labeled Dataset from GitHub:** We incorporated a labeled dataset retrieved from GitHub. This labeled dataset is crucial for the training process, containing Telugu news articles with links, text content, and human-written summaries.

### Data Preprocessing and Working
For our project, the data preprocessing stage is crucial in transforming raw text into a format suitable for analysis and summarization. The detailed preprocessing steps involved in our code include:

1. **Web Scraping with BeautifulSoup**
   - Fetch and parse the webpage content using the `requests` library.
   - Extract the main title (`<h1>` tag) and the body content (`<p>` tags) using BeautifulSoup.

2. **Tokenizing Text with SpaCy**
   - Break down the text into sentences and words using the Telugu language model from SpaCy.

3. **Sentence Tokenization**
   - Split the text into individual sentences using the `sentence_tokenize` function from the `indicnlp` library.

4. **Building Word Frequencies**
   - Record the frequency of each word in the text in a dictionary.

5. **Normalizing Word Frequencies**
   - Normalize the word frequencies by dividing each frequency by the maximum frequency observed.

6. **Vectorizing Sentences**
   - Convert sentences into numerical representations using the Bag-of-Words model (CountVectorizer).
   - Transform the count matrix into a Term Frequency-Inverse Document Frequency (TF-IDF) representation using the TfidfTransformer.

7. **Building a Graph and Calculating PageRank Scores**
   - Construct a graph where each sentence is a node, and edges between nodes are weighted by the similarity (TF-IDF scores).
   - Calculate the importance of each sentence within the graph using the PageRank algorithm.

8. **Generating the Summary**
   - Select the top-ranked sentences to form the summary based on the length of the original content.

### Model Training
Training a text summarization model involves combining state-of-the-art NLP techniques with machine learning. We used the Hugging Face Transformers library and the mBART model, a multilingual sequence-to-sequence model pre-trained on large text corpora.

Steps involved:
1. **Preparing the Dataset**
   - Load a custom dataset from a JSON file, containing pairs of texts and their corresponding summaries.
   - Split the data into training and testing sets.

2. **Model and Tokenizer Initialization**
   - Use the `facebook/mbart-large-cc25` model and its corresponding tokenizer.

3. **Tokenizing the Dataset**
   - Define a preprocessing function to tokenize the text and summary pairs.

4. **Setting Up Training Arguments**
   - Use the `Seq2SeqTrainingArguments` class to define hyperparameters and settings for the training process.

5. **Initializing the Trainer**
   - Initialize the `Seq2SeqTrainer` with the model, training arguments, tokenized datasets, and tokenizer.

6. **Training the Model**
   - Start the training process by calling the `train` method of the trainer.

7. **Saving the Model**
   - Save the trained model and tokenizer for future use.

## Results
After implementing and testing our Telugu text summarization model within the Streamlit interface, the results are as follows:
- **Effectiveness:** The Telugu Text Summarizer demonstrates promising performance, effectively condensing lengthy texts into concise summaries.
- **Coherence and Readability:** The generated summaries exhibit good coherence and readability.
- **Room for Improvement:** Further refinement could enhance performance, especially in handling domain-specific terminology or improving coverage of complex topics.

## Conclusion
The results from our Telugu Text Summarizer are encouraging, reflecting its potential utility in various applications, including content curation, information retrieval, and document summarization. By leveraging advanced NLP techniques and continuously refining the model, we aim to further enhance the summarization quality and broaden its applicability across diverse domains.

## Repository
Find our GitHub repository link: [VarthaGenie_Team_22](https://github.com/raghu-prashanka/VarthaGenie_Team_22)
