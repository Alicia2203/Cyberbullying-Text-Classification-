# Cyberbullying Text Classification Model
Scraped Twitter Data with Twitter API and created a text classifier using supervised machine learning techniques to detect offensive comments which are perceived to result in cyberbullying.

## Introduction
As social media usage grows in popularity among people of all ages, the vast majority of citizens rely on it for day-to-day contact. Although social media is a powerful platform for connecting communities and people, this connectedness also makes it possible for people to attack each other without meeting, even from across the planet. Recently, cyberbullying has been one of the major social media issues. In this context, we undertake the development of a text classifier using supervised machine learning techniques to detect offensive comments which are perceived to result in cyberbullying. We investigate several machine learning algorithms, including logistic regression, Multinomial Naive Bayes and Support Vector Machine (SVM) and determine the best algorithm for our proposed cyberbullying detection model. Our models are trained based on a data set consisting of 1400 tweets text that was scraped from Twitter and manually annotated by us. Furthermore, TF-IDF was used as our feature vector and k-fold cross validation was conducted to evaluate our model.

## Literature Reviews
As social media usage becomes progressively in every age group, a huge majority of people rely on this indispensable medium for daily communication and to freely express opinions on the Internet. Nevertheless, the ubiquity of social media can also lead to an increase of cyberbullying incidents online. Cyberbullying is defined as an aggressive act or behaviour that is carried out using electronic means by a group or an individual repeatedly and over time against a victim who cannot easily defend him or herself (Smith et al., 2008). More often than not, cyberbullying is associated with the use of hate speech online. The significance of identifying and moderating hate speech is evident from the strong association between hate speech and hate crimes (Bayer & Bard, 2020). According to the United Nations (n.d.), “hate speech” refers to offensive discourse targeting an individual or a group based on inherent characteristics, such as race, religion or gender.  

The study of cyberbullying hate speech detection has gained much attention over the past few decades due to its strong negative social impact and the creation of an effective cyberbully detection model has considerable scientific merit. Most recent works involve merging machine learning and natural language processing techniques to detect cyberbullying instances by searching for textual patterns representative of hate speech online. Dinakar et al. (2021), performed experiments with binary and multiclass classifiers such as Naive Bayes, Support Vector Machine (SVM), and Rule-based JRip to detect cyberbullying of youtube comments. Their findings show that the detection of textual cyberbullying can be tackled by building individual topic-sensitive classifiers. Chavan and Shylaja (2015), proposed a model which can predict the probability of comments as comments being offensive to participants, and thus detect whether a comment is a bully or non-bully type. Waseem and Hovy (2016), created a data set of 16k annotated tweets which categorised different types of cyberbullying segmented by the target of the abuse. The researchers used a logistic regression classifier and 10-fold cross validation to test the influence of various features on prediction performance. These works demonstrated the capabilities of using machine learning and natural language processing techniques to detect hate speech.  

## Methodology
To create a machine learning text classification model which can predict whether a tweet consists of hate sentiment, the machine learning classifier has to first be trained with a training data set. Therefore, a data set consisting of 1400 English tweets was collected via Twitter search API with Tweepy in Python. Among the 1400 tweets collected, 1000 tweets are random tweets scraped with the search term “a OR b OR c OR d OR e OR f OR g OR h OR i OR j OR k OR l OR m OR n OR o OR p OR q OR r OR s OR t OR u OR v OR w OR x OR y OR z”. Since hate speech is a real but limited phenomenon, there consist only a small number of hate tweets within the 1000 random tweets collected. Therefore, another 400 potential hate and/or offensive tweets were collected and included in this data set. The terms that were queried for the potential hate tweets are “n*gger”, “whitetr*sh”, “whitepeople”, “b*tch” and “gay”. In addition to tweet content, the dataset also includes metadata such as the tweet id. The data set is then manually annotated by three team members to mitigate annotator bias introduced by any parties. Since hate speech is itself a contested term. For the purpose of this study, we define hate speech as the expression or incitement of hatred towards recipients based on race, gender or sexual orientation. In other words, we consider a tweet to be offensive if it uses a sexist or racial slur, or criticizes or negatively stereotypes a sex or a minority. Hate tweets are labelled as ‘1’ whereas non-hate tweets are marked as ‘0’.  

Before training our classifier with the annotated dataset, it is necessary to deploy data cleaning techniques to remove twitter handles (@user), hyperlinks, special characters, numbers, punctuations, stop words and frequently occurring words that do not give important information to the text in the dataset. After cleaning the text, the text is tokenized into tokens, and each token is lemmatized to its base root form. To gain insights into our dataset, basic data visualisation techniques such as word cloud and bar chart are utilized to understand the common words used in the data set. After exploratory data analysis, our preprocessed data is extracted into features with the Term Frequency-Inverse Document Frequency (TF-IDF) technique to build our classification model. The dataset is then split into training data and test data with the train-test ratio of 80:20. Three supervised machine learning algorithms, namely the Logistic Regression Model, Multinomial Naive Bayes Model and Support Vector Classifier Model were implemented to build our classifiers and the performance of each classifier is evaluated to find the best model. Lastly, the k-fold cross-validation technique is used on the best model among the three classifiers to further evaluate the performance of our model.

## Analysis
Before pre-processing our data, exploratory data analysis is conducted to gain insights into our annotated data set. In figure 4.3.1, a bar chart of the number of tweets counted versus the label (0, 1) is plotted to understand the proportion of the number of hate tweets and non-hate tweets. It is observed that among the 1400 tweets in our annotated data set, 962 tweets are labelled as 0 (non-hate tweets) whereas 438 tweets are labelled as 1 (hate-tweets). In percentage terms, 68.71% of the records are labelled as non-hate tweets and 31.28% of the records are labelled as hate tweets. It is apparent that the classes are imbalanced, however, we do not balance the data because hate speech is a limited phenomenon in reality.  

![image](https://user-images.githubusercontent.com/69787181/187065239-38914ae5-79c1-4362-ada8-20a2401ced06.png)

After pre-processing our data by removing Twitter handles, hyperlinks, special characters, numbers, punctuations, stop words and frequently occurring words that do not give important information, the text is tokenized and lemmatized. A frequency distribution chart is plotted to view the 20 most common tokens in our data set. As shown in figure 4.3.1, the top most common word is our dataset is the word “like”, followed by the word “n*gger” and “b*tch”. As expected, the word “n*gger”, “b*tch” and “whitepeople” are common words in our dataset as they are some of the terms that were queried. Other than the word “n*gger”, “b*tch” and “whitepeople”, it is observed that most of the words have either neutral or positive sentiment, which aligns with the fact that around 69% of our sample dataset is non-hate tweets.  

![image](https://user-images.githubusercontent.com/69787181/187065254-87dd7f40-124d-450c-a3d4-a0f21e11b969.png)

Figure 4.3.3 and Figure 4.3.4 are word cloud plots which present a visual overview of the most important and frequently mentioned keywords for non-hate tweets and hate tweets respectively. In figure 4.3.3, it is observed that most of the words are either neutral or positive, with “people”, “know”, “time”, “think’, “love”, “hope” and “good” being the most frequent ones. Most of the frequent words are compatible with the sentiment which is non hate tweets.  

![image](https://user-images.githubusercontent.com/69787181/187065263-50dc6792-98a6-4942-9c87-276f698f19dc.png)

In figure 4.3.4, we can clearly see that most of the words have negative connotations. With no surprise, the key-word "n*gger", “b*tch”, “whitetr*sh”, and “WhitePeople” is most frequently mentioned as they are some of the terms that were queried. Additionally, it is observed that the word “black”, “stup*d”, “racist”, “sh*t”, “hate”, and “f*ck” are some of the common hateful words mentioned in the tweets detected as hateful. Furthermore, it can be seen that the keyword “American” and “African” are also common terms in the dataset, which are highly likely to be associated with racist hateful tweets that have the keyword “n*gger”, “WhiteTr*sh” and “WhitePeople”.

![image](https://user-images.githubusercontent.com/69787181/187065271-6cf46ade-b81f-489a-98c6-88df864e0cc3.png)

After exploratory data analysis, our preprocessed data is extracted into features with the Term Frequency-Inverse Document Frequency (TF-IDF) technique to transform our data into numerical features usable for our classification model. The dataset is then split into training data and test data with the train-test ratio of 80:20. Among the 1400 records in our dataset, 1120 records is used as training data and 280 records are used as our test data (see figure 4.3.5). Our training data set is then used to train three types of models for our classifier.

![image](https://user-images.githubusercontent.com/69787181/187065276-4ac1a31f-775e-4563-80d3-4d51f2010ba6.png)

## Results & Discussion
Three supervised machine learning algorithms, namely the Logistic Regression Model, Multinomial Naive Bayes Model and Support Vector Classifier Model were implemented to build our classifiers and the performance of each classifier is evaluated. A classification report is generated for each model to measure the quality of predictions from our classification models. The classification report shows the main classification metrics, which include the accuracy, precision, recall and f1-score on a per-class basis. In addition to the classification report, a confusion matrix is also plotted for each classification model.

### Logistic Regression Model
For our logistic regression model, we obtained results of around 85.36% of accuracy, which is considered fairly good (see figure 4.4.1.1). The resulting confusion matrix is shown in figure 4.4.1.2, which shows the values for true positive (TP), false positive (FP), true negative (TN) and false negatives (FN). In view that our classifier aims to detect as many hate tweets as possible so that all potential cyberbullying incidents can be detected, it is essential to look at the FN values. It is observed that out of 280 records, 39 records of hate tweets were wrongly predicted by our logistic regression model as non-hate tweets.

![image](https://user-images.githubusercontent.com/69787181/187065317-643b231f-a063-429d-8e59-f7a796fa1414.png)

![image](https://user-images.githubusercontent.com/69787181/187065323-53f91550-7f95-423e-b902-9a3a2c4e1ad6.png)

### Multinomial Naive Bayes Model
For our Multinomial Naive Bayes model, the test accuracy is around 82.50%, which is slightly less accurate than our logistic regression model (see figure 4.4.2.1). The resulting confusion matrix is shown in figure 4.4.2.2. The FN value for our Multinomial Naive Bayes model is 44. In other words, out of 280 records, 44 records of hate tweets were wrongly predicted as non-hate tweets.

![image](https://user-images.githubusercontent.com/69787181/187065345-e278bef6-dd24-4dc9-8128-ea9615c295a5.png)
![image](https://user-images.githubusercontent.com/69787181/187065348-3bd365e8-444d-4681-80af-b4f36616b01a.png)

### Support Vector Machine Model
For our Support Vector Machine model, the test accuracy is around 86.79%, which is highest among all three models (see figure 4.4.3.1). The resulting confusion matrix is shown in figure 4.4.3.2. The FN value for our Support Vector Classifier model is 33, which means that out of 280 records, only 33 records of hate tweets were wrongly predicted as non-hate tweets. Table 4.4.3.3 shows the performance result summary of all three classifiers.
![image](https://user-images.githubusercontent.com/69787181/187065366-31c8e36c-15b5-4fbc-9878-fd8cb8c12ecb.png)
![image](https://user-images.githubusercontent.com/69787181/187065368-fceb026d-e1da-4417-a99e-8efd495c84ba.png)
![image](https://user-images.githubusercontent.com/69787181/187065376-e556b635-59b3-443e-ab86-1f25077db0d6.png)

In view that our sample data is relatively small, K-fold cross-validation technique was also deployed on our best model, which is the Support Vector Classifier model to determine how well our model would generalise to new datasets. Figure 4.4.3.4 shows that the accuracy score of our Support Vector Classifier model with 10-fold cross-validation is 88.21%. This suggests that our Support Vector Classifier model might perform well in generalising new data sets.

![image](https://user-images.githubusercontent.com/69787181/187065386-f121764c-9e29-489e-b384-327a9181e253.png)

In summary, our Support Vector Classifier Model has the best performance in classifying hateful and non-hateful text for this particular dataset. The second best model is the logistic Regression, followed by the Multinomial Naive Bayes model.

### *Future Improvements*
There is much room for improvement for the development of this cyberbullying text classification model. Given that our models are trained on hate speech targeting African-American and White Americans, our models are likely to be biased and may be more sensitive to anti-African-American and White Americans hate speech, and less sensitive to other types of hate speech. In future work, our data set should be more comprehensive and and non-biased toward certain demographic groups.  

Furthermore, the size of our sample dataset that was used to train our model is relatively small, therefore, the performance of our classifier in classifying real-world data might not perform as well as when it classifies our sample data set. As such, the scale of the training data set can be extended in future work.

## Implication
In terms of how social media analytics can provide solution frameworks for societal issues, we demonstrated in a machine learning text classification model can be created to detect hate speech which may potentially be associated with cyberbullying incidents. Our model was able to classify hate tweets and non-hate tweets with an accuracy score of 88.21%. Hence, it is evident that the use of an efficient cyberbullying detection model can aid in identifying cyberbullying instances. Such a model can be utilised to efficiently detect potential cyberbullying the moment a hate tweet has been posted on social media. This can prevent the escalation of cyberbullying incident and early detection of cyberbullying on social media can be crucial to mitigating the impact on the victims.  

As discussed in the discussions previously, there have been many possible improvements that can be made which have been identified throughout this study. It is without a doubt that there are many things that have been learned to refine our methodologies and executions from this study. This will allow us to become more efficient and effective with our planning as well as coding in later similar assignments to come.

## Conclusion
we experimented with different machine learning algorithms, including logistic regression, Multinomial Naive Bayes and Support Vector Machine (SVM) for hate speech detection and found that the best performance with Support Vector Machine (SVM) as the text classifier, with 88.21% accuracy performance in detecting hate text that may potentially be associated with cyberbullying incidents. However, given that our model is trained on a data set containing hate speech targeting specific demographic groups, our model is likely to be biased and may be more sensitive to a certain type of hate speech. For that reason, further work can include training the model with a dataset that is comprehensive and non-biased toward certain demographic groups. Cyberbullying detection is inherently intricate as the concept of bullying is subjective in nature. Furthermore, Cyberbullying is becoming multi-form and the detection of cyberbullying data is hard to be reached by conventional text analytical techniques. Therefore, further work can be done on integrating multi-form data such as images, videos, and time on social media to cope with the latest type of cyberbullying.











