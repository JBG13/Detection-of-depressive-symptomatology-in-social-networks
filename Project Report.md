# Detection of depressive symptomatology in social networks

## Authors:
* **Jhonatan Barcos Gambaro (100548615)**
* **Andrés Díaz Ruano (100472763)**

## Project description
Currently, with regard to mental health, we are witnessing how much of a problem it is for many people. Moreover, it has been deeply aggravated in the aftermath of the COVID-19 pandemic. According to the WHO, by 2023 almost one in two people in the European Union (46%) had experienced emotional or psychosocial problems in the last 12 months.

These mental health problems also have an economic impact, costing 4% of the GDP of EU countries. Furthermore, the ‘Health at a Glance’ report carried out in 2022 showed that almost one in two young people report having unmet mental health needs. 

The depressive symptomatology of these young people in several EU countries has more than doubled since the pandemic. These data clearly show that there is a huge unmet need with regard to mental health. 

Through this project, we aim to detect depressive symptomatology in social media posts through a Dataset of posts on Reddit providing a reliable method for the detection of these individuals and their treatment.

##  Dataset Description
This dataset is designed for a **binary classification task** aimed at detecting signs of depression in Reddit posts. Each post is labeled as either:

- **1 = Depressed** (originating from mental health-related subreddits)
- **0 = Non-depressed** (from general-interest subreddits)

The posts were collected from six subreddits, grouped by mental health relevance:

- **Depressive sources:**
  - `r/Depression`
  - `r/SuicideWatch`

- **Non-depressive sources:**
  - `r/Teenagers`
  - `r/DeepThoughts`
  - `r/Happy`

Each entry in the dataset contains the following fields:

- `subreddit`: the subreddit where the post was published  
- `title`: the post’s title (often a summary or strong emotional statement)  
- `body`: the main content of the post, where most linguistic cues are found  
- `upvotes`: number of upvotes received (can reflect engagement or sentiment resonance)  
- `created_utc`: UTC timestamp of when the post was created  
- `num_comments`: number of comments received (a proxy for interaction or support)  
- `label`: target variable (0 = non-depressed, 1 = depressed)

Our goal is to explore how different representations capture the emotional and semantic structure of the posts, and assess their usefulness in downstream tasks such as classification or visualization.

# 1. Natural Language Processing, Topic Modeling and Document Vectorization
## 1.1. Preprocessing Pipeline
In this section, we will define the preprocessing carried out on our dataset to adapt it in the best possible way to our project. As a previous step to make the corresponding code, we have made a detailed analysis of the language of our dataset, which we have to take into account that it comes from social network posts, so we find an informal language, with spelling mistakes and/or abbreviations due to the inherent language of social networks. This initial analysis has been key to perform an optimal preprocessing pipeline for our practice.

Below, we detail the preprocessing performed with the aim of cleaning and normalizing the text of our dataset to increase the quality of its representation and subsequent modeling:

First, we have made the concatenation of the title and body of the messages in a single text string for each of the posts. Then, we have casted the text to lowercase, thus reducing the sparsity of the vocabulary. On the other hand, one of the difficulties we have encountered due to the nature of the social networking language is the presence of HTML tags, URLs and user mentions, so we have removed them using regex. Once this initial preprocessing was done, we used the en_core_web_md model to tokenize the vocabulary.

Then, the lemmatization of each token was performed, followed by a final filtering based on the elimination of the following elements:
* Punctuation marks
* Special characters
* Stopwords (coming from SpaCy modified to adapt it to our practice)
* Tokens smaller than 3 characters.
  
Once this preprocessing was done, each document was represented as a clean list of lemmatized tokens prepared for vectorization, which is detailed below.

## 1.2. Document Vectorization Methods
## 1.3. Topic Modeling with LDA

# 2. Machine Learning. Classification using feature extraction or selection techniques



# 3. Implementation of a dashboard using the Python Dash library


# Acknowledgment of authorship


