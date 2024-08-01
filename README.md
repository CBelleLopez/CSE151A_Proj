# CSE151A_Proj

Milestone 1 Update:

We already started on the preprocessing step due to our not being able to properly graph the names feature since those were entered as strings and there is no possible way for us to encode that to numerical values. The sex features were encoded to [0,1] where 0 represented females and 1 represented males. Future preprocessing steps that we could take is dropping fares that cost zero dollars as that would skewer the correlation between fare cost and survival chances. Normalization would have to be implemented for fare and age since the numerical values can increase or decrease drastically which can affect the sensitivity of our algorithm. Nornmalizing them to the same as sex and survival would allow our features to have the same range making it easier to find correlation. 

Milestone 3 Update:

This milestone involved us dealing with data pre-processing. We first tried this out by training a logistic regression model to our data, which involves the general steps of scaling the data and splitting the dataset. We printed out a classification report to see our precision and recall and our model did fairly well with a 79% accuracy. Even printing our scatterplot displayed a check-mark eqsue shape which indicated that the second feature was the least important and growing more important with each subsequent feature. We printed out several scatterplots with their line of best fits, whose discussion and results are included in our notebook. To improve our results, we could maybe have done further normalization and include more datasets to reflect a better prediction on the relations between these features. Our results can be seen within our notebook.



<a target="_blank" href="https://colab.research.google.com/github/CBelleLopez/CSE151A_Proj/blob/main/Project_WriteUp.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Introduction
We chose to make our project about the Titanic because it is one of the most infamous maritime disasters in history resulting in only 706 survivors out of 2,240 total passengers. Our group aims to investigate the factors that could have potentially influenced passenger survival during this famous shipwreck using a data-driven approach. We thought that it would be cool to create a probabilistic model by analyzing the impact of certain attributes on the likelihood of survival and identifying each passenger's class. Specifically, we analyzed the attributes age, sex, the presence of siblings/spouses on board, the presence of parents/children on board, and fare. Through this research, and with a good predictive model, we will be able to enhance our knowledge of survival determinants in large-scale disasters while also honoring the memory of those affected by the tragedy. 

## Methods
  * Data Exploration (Included in the Logistic Regression Model Notebook):
    * For the data exploration section, we read in our dataset and displayed it to see our initial number of features (8) and our initial number of observations (887). Then we did a little bit of preprocessing of our data by dropping the “Name” feature and encoding the “Sex” feature to [0,1] where 0 represented females and 1 represented males. We then calculated some statistical data such as percentile, mean, and standard deviation on our revised dataset. After that, we visualized our data by displaying histograms, pairplots, correlations between columns, and heatmaps. 
  * Preprocessing:
    * (placeholder)
  * Logistic Regression Model:
    * (placeholder)
  * Kmeans Cluster Model: 
    * (placeholder)
  * Decision Tree Model: 
    * (placeholder)
  * Neural Network Model: 
    * (placeholder)

## Results
  * Data Exploration (Included in the Logistic Regression Model Notebook):
    * For the revised dataset, histograms, pairplots, correlations between columns, and heatmaps, the resulting diagrams are displayed within our Logistic Regression Model Notebook. 
  * Preprocessing:
    * (placeholder)
  * Logistic Regression Model:
    * (placeholder)
  * Kmeans Cluster Model: 
    * (placeholder)
  * Decision Tree Model: 
    * (placeholder)
  * Neural Network Model: 
    * (placeholder)

## Discussion

## Conclusion

## Statement of Collaboration
Andrew: Group Member - 

Bryant Quijada: Group Member -  

Clarabelle Lopez: Group Member -  

Fayaz Shaik: Group Member -  

Jordan Phillips: Group Member -  

Owen Lam: Group Member -  
