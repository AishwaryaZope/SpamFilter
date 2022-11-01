# Spam_filter_for_Quora_questions
The objective of this project is to build a Deep Learning Model for detecting the spam questions on Quora.

Data: The provided data consists of 10,48,576 questions from Quora divided in to spam and not spam categories. These questions are provided in an excel file.

Procedure: 
Step 1: Cleaning the data to remove outliers, puntuation marks and then converting the sentenses in to lowercase.

Step 2: tokenising the words in the sentenses using tokenizer. And using GloVe vector to vectorize the words.

Step 3: extract the vectors that are required for data set of words.

Step 4: Splitting the processes data in to Train and test sets

Step 4: Build the 1D Convolution model

 
The performance of the model can be further improved by correcting the spelling mistakes, unnecessary number and symbols.
