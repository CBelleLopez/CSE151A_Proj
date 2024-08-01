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
    * Data Exploration Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Preprocessing:
    * For the preprocessing section, we built upon the basic preprocessing that we did in the data exploration section of each of our four models. For the Logistic Regression Model (which also contains our data exploration), we transformed our modified dataset and normalized the values of our features, and we used ‘Survived’ as our target variable when splitting our dataset into train and test datasets. For the Kmeans Cluster Model, we created a new feature labeled ‘Family_Size’, we dropped the features 'Siblings/Spouses Aboard' and 'Parents/Children Aboard', and we used ‘Pclass’ as our target variable when splitting our dataset into train and test datasets. For the Decision Tree Model, we created a new feature labeled ‘Family_Size’, and we used ‘Pclass’ as our target variable when splitting our dataset into train and test datasets. For the Neural Network Model, we created two new features labeled ‘Family Size’ and ‘Travel Alone’, we transformed our modified dataset and normalized the values of our features, we one hot encoded our ‘Pclass’ feature into three distance features which were ‘1’, ‘2’, and ‘3’ (each number represents the passengers class where 3 is the lower class, 2 is the middle class, and 1 is the upper class), and we used our three distinct Pclass features (‘1’, ‘2’, and ‘3’) as our target variables.
    * Note: Each model does different major preprocessing, but the links for other models are displayed in the subsection of each model.
  * Logistic Regression Model:
    * For the Logistic Regression Model, we utilized logistic regression and applied it to our train/test datasets. We then made predictions on our train and test data and checked our model’s accuracy. We also printed out a classification report on our test data, printed out our feature coefficients using a scatterplot, and printed out several scatterplots comparing different features with our target variable ‘Survived’ using their line of best fits.
    * Logistic Regression Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Kmeans Cluster Model: 
    * (placeholder)
    * Kmeans Cluster Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/KMeans_Cluster_Model_Project_WriteUp.ipynb
  * Decision Tree Model: 
    * (placeholder)
    * Decision Tree Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Decision_Tree_Model_Project_WriteUp.ipynb
  * Neural Network Model: 
    * (placeholder)
    * Neural Network Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Neural_Network_Model_Project_WriteUp.ipynb

## Results
  * Data Exploration (Included in the Logistic Regression Model Notebook):
    * For the revised dataset, histograms, pairplots, correlations between columns, and heatmaps, the resulting diagrams are displayed within our Logistic Regression Model Notebook.
    * Data Exploration Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Preprocessing:
    * There were not many results for the major preprocessing since there was not much output during major preprocessing, but any resulting output from major preprocessing is displayed within each Model’s Notebooks.
    * Note: Each model does different major preprocessing, but the links for other models are displayed in the subsection of each model.
  * Logistic Regression Model:
    * For our logistic regression model, around "0.45" was our training error and around "0.44" was our testing error. When we printed out a classification report to see our precision and recall, our model had around 79% accuracy on the training data and around 0.82% accuracy on the test data. Even printing our scatterplot displayed a check-mark eqsue shape. Taking a look at the coefficients, Pclass (Passenger class) seems to have a negative coefficient of around -1.97, sex has a coefficient of around -2.40, age has a coefficient of -1.86, siblings/spouses aboard has a negative coefficient of -1.58, parents/children aboard has a negative coefficient of -0.42, and fare has a coefficient of 0.55. For the classification report, coefficient scatterplot, and logistic regression plots, the resulting diagrams are displayed within our Logistic Regression Model Notebook.
    * Logistic Regression Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Kmeans Cluster Model: 
    * (placeholder)
    * Kmeans Cluster Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/KMeans_Cluster_Model_Project_WriteUp.ipynb
  * Decision Tree Model: 
    * (placeholder)
    * Decision Tree Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Decision_Tree_Model_Project_WriteUp.ipynb
  * Neural Network Model: 
    * (placeholder)
    * Neural Network Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Neural_Network_Model_Project_WriteUp.ipynb

## Discussion

## Conclusion

## Statement of Collaboration
Andrew: Group Member - 

Bryant Quijada: Group Member -  

Clarabelle Lopez: Group Member -  

Fayaz Shaik: Group Member -  

Jordan Phillips: Group Member -  

Owen Lam: Group Member -  
