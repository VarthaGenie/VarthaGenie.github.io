# Vartha Genie: Building a Telugu News Article Summarizer

**Authors:** Raghu Prashanka, Charan Sai, Badreesh, Dhanush Reddy, Sai Varun

---

## Problem Statement

This project aims to develop a Natural Language Processing model capable of summarizing Telugu news articles. The model will take a Telugu news article URL as input and output a concise and informative summary of the article content.

## Motivation

There is a growing demand for consuming news content in regional languages like Telugu. This project seeks to bridge this gap by creating a Telugu news article summarizer. This model will empower Telugu speakers to:

- Quickly grasp the key points of news articles without having to read the entire piece.
- Stay informed about current events in their native language.
- Enhance their overall engagement with Telugu news media.
- Convenience at fingertips where users can just copy-paste the link and get the summary.
- This is extremely useful for students, professionals, and general readers of Telugu background needing quick news digests.

---

## Technical Approach

### Data Collection

To build a robust Telugu news article summarizer, we used a large corpus of Telugu news articles. This data served as the foundation for training the NLP model. Here's a breakdown of the data collection process:

- **Integration with Labeled Dataset from GitHub:** We incorporated a labeled dataset from GitHub containing Telugu news articles with links, text content, and corresponding human-written summaries. This dataset improves training efficiency and enhances summary quality by providing clear examples of article-summary pairs.

### Data Preprocessing and Working

Data preprocessing is crucial in transforming raw text into a format suitable for analysis and summarization. Key steps include:

1. **Web Scraping with BeautifulSoup:** Fetch and parse webpage content to extract the main title and body content.
2. **Tokenizing Text with Indic NLP library:** Use the Telugu language model from Indic NLP library to process the text.
3. **Sentence Tokenization:** Split the text into individual sentences using the `sentence_tokenize` function from the Indic NLP library.
4. **Building Word Frequencies:** Record the frequency of each word in the text to determine sentence significance.
5. **Normalizing Word Frequencies:** Normalize word frequencies by dividing each frequency by the maximum frequency observed.
6. **Vectorizing Sentences:** Convert sentences into numerical representations using the Bag-of-Words model and transform into a TF-IDF representation.
7. **Building a Graph and Calculating PageRank Scores:** Construct a graph with sentences as nodes, weighted by similarity (TF-IDF scores), and calculate importance using PageRank.
8. **Generating the Summary:** Select top-ranked sentences to form the summary based on the original content's length.

---

## Model Training

### Preparing the Dataset

- Load a custom dataset from a JSON file containing text-summary pairs.
- Split the dataset into training and testing sets for evaluation on unseen data.

### Model and Tokenizer Initialization

- Use the `facebook/mbart-large-cc25` model, ideal for multilingual tasks.
- Load the corresponding tokenizer for text preprocessing.

### Tokenizing the Dataset

- Convert raw text into a format suitable for the model.
- Define a function to tokenize text and summary pairs.
- Truncate text and summaries to a maximum length and apply padding for consistent input sizes.

### Setting Up Training Arguments

- Specify hyperparameters and settings with the `Seq2SeqTrainingArguments` class.
- Key parameters include learning rate, batch size, number of epochs, and evaluation strategy.

### Initializing the Trainer

- Use the `Seq2SeqTrainer` class to manage training, evaluation, and saving.
- Initialize the trainer with the model, training arguments, tokenized datasets, and tokenizer.

### Training the Model

- Start the training process by calling the `train` method of the trainer.
- The model learns to generate summaries from input text and is periodically evaluated on the test set.

### Saving the Model

- Save the trained model and tokenizer to a specified directory.
- Ensure they can be loaded and used for generating summaries on new data.

---

## The Journey

Initially, we considered translating the Telugu text to English, performing the summarization, and then translating the summary back to Telugu. This approach was tempting due to the extensive availability of NLP tools and resources in English, which are often more advanced and well-supported compared to those for Telugu. However, this method has several significant disadvantages. 

First, the translation process can introduce errors and distortions in the text, affecting the accuracy and integrity of the summarization. Each translation step—both to and from English—risks losing nuanced meanings and cultural context inherent in the original Telugu text. Additionally, the double translation process is computationally expensive and time-consuming, which could lead to inefficiencies in processing large volumes of text. Therefore, we decided to directly work with Telugu text, leveraging specific models and resources for Indian languages to ensure accuracy, efficiency, and cultural relevance in our summarizations.

After showing this approach to our mentor Prof. Nidhi Goyal, we were advised to train a model on an existing dataset of text and summarized text. Doing so would yield more accurate results and summaries. We then utilized a pre-trained model from Hugging Face and trained the model as detailed in the "Model Training" section.

---

## Results

After implementing and testing our Telugu text summarization model within the Streamlit interface, the results are as follows:

- Evaluated the effectiveness of the summarizer using diverse web pages with different types of content.
- Assessed how well the summaries encapsulate key information present in the original content.
- Analyzed whether the generated summaries accurately capture main ideas and crucial details from the original text.

---

## Findings

Our Telugu News Article Summarizer project findings include:

- Effectively captured the main points of Telugu news articles, providing summaries that retain essential information.
- Generated summaries maintained logical flow and coherence, making them easy to understand and follow.
- Performed well across various news domains, including politics, sports, technology, and entertainment, and handled varying lengths of input texts.
- Can be further fine-tuned to cater to specific user preferences, such as focusing on certain topics or adjusting summary lengths.
- Contributed to the development of NLP tools for Telugu, a relatively less-resourced language, enhancing the availability of digital tools for Telugu speakers.
- Can be integrated into news websites and apps, offering an added feature for users to get quick summaries directly on the platforms they use.

---

## Observations and Insights

- **Effectiveness:** Demonstrated promising performance, effectively condensing lengthy texts into concise summaries.
- **Coherence and Readability:** Generated summaries exhibited good coherence and readability, making them comprehensible to readers.
- **Room for Improvement:** Further refinement could enhance performance, such as handling domain-specific terminology or improving coverage of complex topics.

---

## Conclusion

In conclusion, the results from our Telugu Text Summarizer are encouraging, reflecting its potential utility in various applications, including content curation, information retrieval, and document summarization. By leveraging NLP techniques and continuously refining the model, we aim to further enhance the summarization quality and broaden its applicability across diverse domains.

---

![Thank You](link_to_thankyou_image)

## GitHub Repository

Find our GitHub repository [here](https://github.com/raghu-prashanka/VarthaGenie_Team_22).
