**Description of Files**

**model:** contains OneVsRest Logistic Regression model and TF-IDF vector used for training models  
**reddit:** contains 12 pickled Reddit subreddits, files in the format subreddit_numberofcomments.pkl  
**Binary_Classification_Models.ipynb:** Jupyter Notebook walkthrough of binary classification models  
**CNN_word2vec.py:** CNN model built using keras and a word2vec embedding layer  
**One_vs_Rest_Model.ipynb:** Jupyter Notebook walkthrough of OneVsRest model with Logistic Regression  
**PRAW_comment_scraper.py:** code used to scrape Reddit using PRAW  
**Topic_Modeling_NMF_LDA.ipynb:** Jupyter Notebook walkthrough of topic modeling using sk-learn's NMF and LDA  
**data_sample.csv:** sample of data used to train model (Wikipedia talk page edits provided by Google / Jigsaw --> https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)  
**label_encoder.pkl:** pickled MultiLabelBinarizer from sk-learn used to numberically encode classes (clean, toxic, severe toxic, obscene, threat, insult, and identity hate)  
**predict_CNN.py:** code used to make predictions on new comments using CNN model  
**predict_OvR.py:** code used to make predictions on new comments using OneVsRest with Logistic Regression model  
**preprocess.py:** preprocessing of text for both CNN and OvR models  

# Toxic Waste Removal: Classifying Harmful Comments on the Web  
  
By Alex Smith  
  
### "The Internet, in general, I find troubling. The anonymity has made us all meaner and dumber. This thing that was supposed to bring us closer together, I see it doing the opposite." - Aaron Sorkin  
  
**Overview: Problem and Background**  
  
When I was a sophomore in high school, I remember reading about two 13-year-old who ended their lives within the same week. Both had been severely cyberbullied. That is when I learned how dark and dangerous the web can be--especially for teens and adolescents, or anyone perusing the internet in a vulnerable state of mind. This begs the question, **how can we make the internet a safer place?** My approach was to build a classification model to label comments as belonging to at least one of the following seven classes: clean, toxic, severe toxic, obscene, insult, threat, and identity hate.

**Data**

The data I worked with consisted of ~150,000 comments of Wikipedia talk page edits, provided by Google’s Jigsaw through a recent Kaggle competition. The comments were labeled as belonging to one of the following categories: toxic, severe toxic, obscene, insult, threat, and identity hate. An absence of any of those six labels indicated that a particular comment was clean. The distribution of classes in the data was unbalanced, with ~10% of the comments identified as a harmful comment (of any subclass of harm) and ~90% clean comments.

**Project Design**

A central challenge of this project was the fact that it is a multi-label classification problem. The classes that a comment may belong to are not mutually exclusive. For example, a harmful comment may be toxic as well as obscene as well as an instance of identity hate. For my MVP though, I started with a simplified version of the problem: binary classification identifying a comment as harmful or clean. I built a logistic regression model, a decision tree, and a random forest after balancing the classes (by undersampling) and preprocessing the text. Preparing the text for the models included tokenizing and lemmatizing the text, creating bigrams and trigrams, and vectorizing the text using TF-IDF vectorizer. Of the binary classification models, the logistic regression model performed the best.

With this result, I built a One vs. Rest multi-label classifier with logistic regression. Additionally, I trained a CNN network with a word2vec pre-trained embedding layer, which performed the best of all of the models I created. With my multi-class models built, I scraped 12 Reddit subreddits and classified the the comments using both models.

Beyond the classification aspect of the project, I was interested in exploring the actual content of the comments. I performed topic modeling on the harmful comments subset of the full dataset using LDA and NMF in Scikit-Learn.

**Tools**

The main tools that I used in addition to Python, Pandas, PyCharm, and Jupyter Notebook were Keras, Scikit-Learn, and a P3 EC2 instance on AWS. Other libraries I used include NumPy, pickle, Matplotlib, Seaborn, Gensim, and PRAW.

**Algorithm / Results**

In total, I built, trained, tested, and evaluated five classification models, three of which were binary classification models and two of which were multi-class. The results of the binary and multi-class models can be viewed below.

*Results of Binary Classification Models (Harmful or Clean)*  
  
<img width="517" alt="binary_results" src="https://user-images.githubusercontent.com/34464435/42338692-86abc78e-803f-11e8-8d19-988efefa61fd.png">

*Results of Multi-class Models*  
  
<img width="518" alt="multi-class_results" src="https://user-images.githubusercontent.com/34464435/42338705-911ecf68-803f-11e8-8b61-79b8107e955d.png">

**Future Direction**

There are four main directions I would like to explore further with this project in the future. First of all, I would like to further explore the subclasses of harmful comments because there is space to improve the model performance and impact further. Secondly, I would love to put my model(s) to work in decreasing cyberbullying and making the internet a safer place. Additionally, it would be interesting to explore creating a chatbot to de-escalate users writing harmful comments in real time, especially particularly mean-spirited and cruel comments. Finally, I would like to create an interactive dashboard of plots showing trends between classification models and datasets or sites. This would give more insight about the problem and the strengths and weaknesses of different models in regards to the problem of harmful comments.

**Next Time**

If I did this project again, the main thing I would do differently is stick to the PEP-8 style guide more strictly. I think there is a very high ROI for writing beautiful, maintainable, and easily understandable code, which I feel I did well, but could have done even better. I will focus on this for my next project.

**Resources I found helpful**

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://motherboard.vice.com/en_us/article/539qaa/someone-quantified-which-subreddits-are-the-most-toxic
https://phys.org/news/2017-07-net-neutrality-day-action-internet.html

**Contact**

email: datagrlxyz @ gmail dot com
twitter: @datagrlxyz
blog: www.datagrl.xyz
