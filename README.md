# Anime Recommender
### Suggest animes based on your historic 

# 1.0 Context

## 1.1 What are animes?
Anime (Japanese: アニメ) is a Japanese term for animation. 
Outside of Japan and in English, anime refers specifically 
to animation produced in Japan.However, in Japan and in Japanese,
anime (a term derived from a shortening of the English word animation) describes all animated works, regardless 
of style or origin. Animation produced outside of Japan with 
similar style to Japanese animation is commonly referred to as 
anime-influenced animation.

# 2.0 The Challenge
Find new animes or other content to watch in next may be a problem
sometimes. Even if you look at the popular series and brand-new ones
it's hard to find something that fits exactly your taste, this project
has as main goal to suggest a list of series to watch based on your 
watch history.

# 3.0 The Solution
In this project i managed to address the challenge by developing a 
ensemble of classification model deployed on a webpage. The ensemble
has an average precision that ranges from 70.88% to 74.88% to suggest
a series that fit's your taste.

## 3.1 Data Source
All data used in this project was extracted by MyAnimeList API.
To know more about this API consult: https://myanimelist.net/apiconfig/references/api/v2

## 3.2 Data Storage
After extraction the data is storage into a MySQL Database on AWS RDS

## 3.3 Labeling
The labels for the anime series was made using Google Sheets to 
visualize the tittle, mean score, number os episodes and picture and
based on that labeling with 1 "I would like" or 0 "I wouldn't like".
Since this method is very tough to do, I used Active Learning as a 
strategy to improve the model results with fewer labels.

## 3.4 Modeling
- TFDFVectorized was applied on text features;
- To scale numeric features was chosed MaxAbsScaler;
- Was used a LightGBM and Logistic Regression Ensemble with a 3/7 
balance.

## 3.5 App list
This project was finished building a web application with Flask to 
shows a list with the anime series with best scores on the model as 
a suggestion. All project was containerized with Docker and uploaded
in Heroku.

# 4.0 Next Steps

4.1 Develop an app that intakes labels from different users to train 
the model and deploy a different result for each one.

4.2 Build a endpoint and improve de app list with better design.

4.3 Build a model retraining pipeline.
