# Vartha Genie: Building a Telugu News Article Summarizer

## Authors
Raghu Prashanka, Charan Sai, Badreesh, Dhanush Reddy, Sai Varun

![Image 1]images/image1)

## Problem Statement
This project aims to develop a Natural Language Processing model capable of summarizing Telugu news articles. The model will take a Telugu news article URL as input and output a concise and informative summary of the article content.

## Motivation
There is a growing demand for consuming news content in regional languages like Telugu. This project seeks to bridge this gap by creating a Telugu news article summarizer. This model will empower Telugu speakers to:
- Quickly grasp the key points of news articles without having to read the entire piece.
- Stay informed about current events in their native language.
- Enhance their overall engagement with Telugu news media.
- Convenience at fingertips where users can just copy paste the link and get the summary.
- This is extremely useful for students, professionals, and general readers of Telugu background needing quick news digests.

## Technical Approach
While working on this project, we did the following steps:

### Data Collection
To build a robust Telugu news article summarizer, we took a large corpus of Telugu news articles. This data served as the foundation for training the NLP model. Here's a breakdown of the data collection process:
- **Integration with Labeled Dataset from GitHub:** We have incorporated a labeled dataset retrieved from GitHub. This labeled dataset is crucial for the training process. It likely contains Telugu news articles with links, text content, and each article has a corresponding human-written summary.
- **Advantages of Labeled Data:** 
  - Improved Training Efficiency: Labeled data provides clear examples of article-summary pairs, allowing the model to learn the relationship between the full article and its concise summary.
  - Enhanced Summary Quality: By training on human-written summaries, the model can learn to generate summaries that are not only factually accurate but also grammatically correct and stylistically similar to human-written summaries.

![Image 2]images/image2)

### Data Preprocessing and Working
For our project, the data pre-processing stage is crucial in transforming raw text into a format suitable for analysis and summarization. Let's delve into the detailed preprocessing steps involved in our code:

1. **Web Scraping with BeautifulSoup:**
   - The first step involves fetching and parsing the webpage content. We use the requests library to make an HTTP GET request to the provided URL. If the request is successful, the HTML content is parsed using BeautifulSoup, extracting the main title (`<h1>` tag) and the body content (`<p>` tags). This text extraction is crucial as it sets the foundation for our summarization process.

2. **Tokenizing Text with Indic NLP library:**
   - Once we have the raw text, the next step is tokenization, which involves breaking down the text into sentences and words. We load the Telugu language model from Indic NLP library and use it to process the text. This step is essential for analyzing the text structure and preparing it for frequency analysis.

3. **Sentence Tokenization:**
   - For the summarization task, we split the text into individual sentences using the `sentence_tokenize` function from the indicnlp library. This function is specifically designed to handle sentence tokenization for Indian languages, including Telugu.

4. **Building Word Frequencies:**
   - To identify the most important sentences, we first need to understand the frequency of each word in the text. We iterate over the tokenized words, excluding stopwords (though in this case, we assume no stopwords are used for Telugu). Each word's frequency is recorded in a dictionary. This frequency count helps in determining the significance of each word in the context of the entire document.

5. **Normalizing Word Frequencies:**
   - We normalize the word frequencies by dividing each frequency by the maximum frequency observed. This normalization helps in scaling the word frequencies, ensuring that the highest frequency words do not disproportionately influence the sentence scoring.

6. **Vectorizing Sentences:**
   - The next step involves converting sentences into numerical representations using the Bag-of-Words model (CountVectorizer). We then transform this count matrix into a Term Frequency-Inverse Document Frequency (TF-IDF) representation using the TfidfTransformer. The TF-IDF matrix highlights the importance of words relative to the document.

![Image 3]images/image3)

7. **Building a Graph and Calculating PageRank Scores:**
   - We construct a graph where each sentence is a node, and edges between nodes are weighted by the similarity (TF-IDF scores). Using the PageRank algorithm, we calculate the importance of each sentence within the graph. Sentences with higher scores are deemed more significant for the summary.

![Image 4]images/image4)

8. **Generating the Summary:**
   - Finally, we select the top-ranked sentences to form the summary. The number of sentences included in the summary is determined based on the length of the original content, ensuring a concise and relevant summary.

### Model Training
Training a text summarization model is an exciting journey that combines state-of-the-art NLP techniques with the power of machine learning. We'll walk through the steps involved in training a text summarization model using the Hugging Face Transformers library, specifically focusing on the model training part. We'll be working with a dataset of Telugu text summaries and the mBART model, a multilingual sequence-to-sequence model pre-trained on large text corpora.

#### Preparing the Dataset
Before diving into the model training, let's briefly touch on the data preparation. We load a custom dataset from a JSON file, which contains pairs of texts and their corresponding summaries. After loading the data, we split it into training and testing sets, ensuring our model can be evaluated on unseen data.

#### Model and Tokenizer Initialization
We use the `facebook/mbart-large-cc25` model, which is well-suited for multilingual tasks, including text summarization. The model is loaded alongside its corresponding tokenizer, which handles text preprocessing.

![Image 5]images/image5)

#### Tokenizing the Dataset
Tokenization is a critical preprocessing step where raw text is converted into a format suitable for the model. We define a preprocessing function to tokenize the text and summary pairs. Each text and summary is truncated to a maximum length, and padding ensures consistent input sizes.

#### Setting Up Training Arguments
Training a model involves specifying various hyperparameters and settings that control the training process. We use the `Seq2SeqTrainingArguments` class to define these settings. Key parameters include the learning rate, batch size, number of epochs, and evaluation strategy. These settings ensure that the model is trained efficiently and can be evaluated periodically.

#### Initializing the Trainer
The `Seq2SeqTrainer` class from the Transformers library simplifies the training process by managing the training loop, evaluation, and saving of the model. We initialize the trainer with the model, training arguments, tokenized datasets, and tokenizer. This setup prepares the model for the training process.

#### Training the Model
With everything set up, we start the training process by calling the `train` method of the trainer. This method initiates the training loop, where the model learns to generate summaries from the input text. The training progress is periodically evaluated on the test set to monitor performance.

#### Saving the Model
After training, it's crucial to save the model and tokenizer for future use. We save the trained model and tokenizer to a specified directory, ensuring that they can be loaded and used for generating summaries on new data.

![Image 6]images/image6)

### The Journey
Initially, before the development of previously discussed algorithm, we considered translating the Telugu text to English, performing the summarization, and then translating the summary back to Telugu. This approach was tempting due to the extensive availability of NLP tools and resources in English, which are often more advanced and well-supported compared to those for Telugu. 

However, this method has several significant disadvantages. First, the translation process can introduce errors and distortions in the text, affecting the accuracy and integrity of the summarization. Each translation step—both to and from English—risks losing nuanced meanings and cultural context inherent in the original Telugu text. 

Additionally, the double translation process is computationally expensive and time-consuming, which could lead to inefficiencies in processing large volumes of text. Therefore, we decided to directly work with Telugu text, leveraging specific models and resources for Indian languages to ensure accuracy, efficiency, and cultural relevance in our summarizations.

Therefore, we found another approach to proceed with our project as seen above.

And then the other phase began when we have shown this approach to our mentor Prof. Nidhi Goyal. She then asked us to train a model on an existing dataset of text and summarized text. Doing so would yield more accurate results and summaries. We began focusing on this and utilized a pre-trained model from Hugging Face and trained the model as seen above in the “Model Training” section.

## Results
After implementing and testing our Telugu text summarization model within the Streamlit interface, the results are as follows:
- To evaluate the effectiveness of our Telugu Text Summarizer, we utilized a diverse set of web pages containing different types of content, ranging from news articles to blog posts and educational materials. 
- We assessed how well the summaries encapsulate the key information present in the original content. Higher coverage indicates better summarization quality.
- We have also analyzed whether the generated summaries accurately capture the main ideas and crucial details from the original text.

![Image 7]images/image7)
![Image 8]images/image8)
![Image 9]images/image9)

### Findings
Our Telugu News Article Summarizer project findings include:
- The model effectively captures the main points of Telugu news articles, providing summaries that retain essential information.
- The generated summaries maintain logical flow and coherence, making them easy to understand and follow.
- The summarizer also performs well across various news domains, including politics, sports, technology, and entertainment. Also, the model can handle varying lengths of input texts.
- The summarizer can be further fine-tuned to cater to specific user preferences, such as focusing on certain topics or adjusting summary lengths.
- The project contributes to the development of NLP tools for Telugu, a relatively less-resourced language, enhancing the availability of digital tools for Telugu speakers.
- The summarizer can be integrated into news websites and apps, offering an added feature for users to get quick summaries directly on the platforms they use.

### Observations and Insights
Upon analyzing the results, we observed the following:
- **Effectiveness:** Our Telugu Text Summarizer demonstrates promising performance, effectively condensing lengthy texts into concise summaries.
- **Coherence and Readability:** The generated summaries exhibit good coherence and readability, making them comprehensible to readers.
- **Room for Improvement:** While the summarizer performs considerably well overall, there are areas where further refinement could enhance its performance, such as handling domain-specific terminology or improving coverage of complex topics.

## Conclusion
In conclusion, the results from our Telugu Text Summarizer are encouraging, reflecting its potential utility in various applications, including content curation, information retrieval, and document summarization. By leveraging NLP techniques and continuously refining the model, we aim to further enhance the summarization quality and broaden its applicability across diverse domains.

![Image 10]images/image10)

Find our Github repository link: [Github Repository](https://github.com/raghu-prashanka/VarthaGenie_Team_22)
