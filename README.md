# Stack-Overflow-Tag-Prediction

Stack Overflow is the largest, most trusted online community for developers to learn, share their programming knowledge, and build their careers.

Stack Overflow is something that every programmer uses one way or another. Each month, over 50 million developers come to Stack Overflow to learn, share their knowledge, and build their careers. It features questions and answers on a wide range of topics in computer programming. The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. As of April 2014, Stack Overflow has over 4,000,000 registered users, and it exceeded 10,000,000 questions in late August 2015. Based on the type of tags assigned to questions, the top eight most discussed topics on the site are Java, JavaScript, C#, PHP, Android, jQuery, Python, and HTML.

## Problem Statement
Suggest the tags based on the content that was there in the question posted on Stackoverflow.

## About the data:
You can download the data from: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data

## Useful Resources:

Youtube: https://youtu.be/nNDqbUhtIRg 
Research paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tagging-1.pdf 
Research paper: https://dl.acm.org/citation.cfm?id=2660970&dl=ACM&coll=DL

### Phase 1: Data Collection and Data Cleaning:

You can download the data from the above-mentioned Kaggle link.
Text data preprocessing like removal of stop words, stemming, etc using NLP libraries(Ex: NLTK). 
Handle missing values, outliers, and data inconsistencies through imputation or removal.
Explore the data, structure, column meanings, and table relationships.

### Phase 2: Data analysis:

Data-set level and output-variable analysis
Perform univariate Feature Analysis. 
Perform multivariate Feature analysis. 
For each of the above analyses and plots, please provide a point-wise summary of key observations at the end.

### Phase 3: Model Building, Debugging, and Feature Engineering:

Base-line model and metrics
Try out incrementally more complex models as long as they satisfy real-world constraints and requirements.
Hyper-parm tune your models using various approaches including black-box optimization methods.
Thorough Error analysis on your best models. Key observations and how can this analysis help you better design new features or models (in the next phase).
Robustness of your best models.
What are the pros and cons of the models you used? When and where should each of them be used in real-world applications? 
Using the error analysis performed above, design new features.
Implement the advanced feature encoding and engineering methods described in the literature survey and measure their impact.
Model interpretability (instance level and overall-model level) and summary of observations.


### Phase 4: Model Deployment:

Deploy the Model In any of the cloud platforms(AWS, GCP, etc)
Showcase a working demo of your system that is easy for an end user to interact with and use.


Guidelines:
the objectives the mentioned above are to show you a glimpse of what are the possible tasks that can be performed but the whole/excat tasks, we will be discussing the appropriate tasks that can be performed during our first call.
Before we start our projects please go through the data set try downloading it and if time permits please check out the features and their meaning
Please revise concepts of pandas, numpy, and matplotlib and any other python library
There might be some tasks which might need to use new library which will be discussed and resources will be provided.
