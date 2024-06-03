# Vartha Genie: Building a Telugu News Article Summarizer

**Authors**: Raghu Prashanka, Charan Sai, Badreesh, Dhanush Reddy, Sai Varun

![ ](VarthaGenie.github.io/image1)

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
### Data Collection
To build a robust Telugu news article summarizer, we took a large corpus of Telugu news articles. This data served as the foundation for training the NLP model.

![ ](VarthaGenie.github.io/image2)

#### Integration with Labeled Dataset from GitHub
We have incorporated a labeled dataset retrieved from GitHub. This labeled dataset is crucial for the training process. It likely contains Telugu news articles with links, text content and each article has a corresponding human-written summary.

### Data Preprocessing and Working
The data pre-processing stage is crucial in transforming raw text into a format suitable for analysis and summarization. The steps involved include:

1. **Web Scraping with BeautifulSoup**
2. **Tokenizing Text with Indic NLP library**
3. **Sentence Tokenization**
4. **Building Word Frequencies**
5. **Normalizing Word Frequencies**
6. **Vectorizing Sentences**
7. **Building a Graph and Calculating PageRank Scores**
8. **Generating the Summary**

![ ](VarthaGenie.github.io/image3)

### Model Training
Training a text summarization model involves the following steps using the Hugging Face Transformers library, focusing on the mBART model:

1. Preparing the Dataset
2. Model and Tokenizer Initialization
3. Tokenizing the Dataset
4. Setting Up Training Arguments
5. Initializing the Trainer
6. Training the Model
7. Saving the Model

![ ](VarthaGenie.github.io/image4)

## The Journey
Initially, we considered translating the Telugu text to English, summarizing, and then translating back to Telugu. However, this approach introduced errors and inefficiencies. Therefore, we decided to directly work with Telugu text, leveraging specific models and resources for Indian languages.

Our mentor, Prof. Nidhi Goyal, suggested training a model on an existing dataset of text and summarized text for more accurate results. We utilized a pre-trained model from Hugging Face and trained the model as detailed in the “Model Training” section.

## Results
After implementing and testing our Telugu text summarization model within the Streamlit interface, the results are as follows:

![ ](VarthaGenie.github.io/image5)
![ ](VarthaGenie.github.io/image6)

## Findings
Our Telugu News Article Summarizer project findings include:

- The model effectively captures the main points of Telugu news articles, providing summaries that retain essential information.
- The generated summaries maintain logical flow and coherence, making them easy to understand and follow.
- The summarizer performs well across various news domains and handles varying lengths of input texts.
- The summarizer can be further fine-tuned to cater to specific user preferences.
- The project contributes to the development of NLP tools for Telugu, enhancing the availability of digital tools for Telugu speakers.
- The summarizer can be integrated into news websites and apps.

## Observations and Insights
- **Effectiveness**: Our Telugu Text Summarizer demonstrates promising performance, effectively condensing lengthy texts into concise summaries.
- **Coherence and Readability**: The generated summaries exhibit good coherence and readability.
- **Room for Improvement**: Further refinement could enhance performance, such as handling domain-specific terminology or improving coverage of complex topics.

## Conclusion
The results from our Telugu Text Summarizer are encouraging, reflecting its potential utility in various applications. By leveraging NLP techniques and continuously refining the model, we aim to further enhance the summarization quality and broaden its applicability across diverse domains.

![Thank You Summary Image](path_to_thankyou_image)

## Repository Link
Find our Github repository link: [GitHub Repository](https://github.com/your_repository_link)
