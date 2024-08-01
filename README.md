# CSE151A_Proj

## Introduction
We chose to make our project about the Titanic because it is one of the most infamous maritime disasters in history resulting in only 706 survivors out of 2,240 total passengers. Our group aims to investigate the factors that could have potentially influenced passenger survival during this famous shipwreck using a data-driven approach. We thought that it would be cool to create a probabilistic model by analyzing the impact of certain attributes on the likelihood of survival and identifying each passenger's class. Specifically, we analyzed the attributes age, sex, the presence of siblings/spouses on board, the presence of parents/children on board, and fare. Through this research, and with a good predictive model, we will be able to enhance our knowledge of survival determinants in large-scale disasters while also honoring the memory of those affected by the tragedy. 

## Methods
  * Data Exploration (Included in the Logistic Regression Model Notebook):
    * For the data exploration section, we read in our dataset and displayed it to see our initial number of features (8) and our initial number of observations (887).
    * Before we decided to explore our data fully, we decided to preprocess data, with methods that are discussed in the preprocessing section. After this step, we explored our data by printing out several eye-catching graphics which were discussed during lecture. We also chose to print out graphics like histograms, pairplots, and correlation coefficients. We then calculated some statistical data such as percentile, mean, and standard deviation on our revised dataset. 
    * Data Exploration Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Preprocessing:
    * For preprocessing our dataset, we built upon the basic preprocessing which included dropping the 'Name' feature and convert the 'Gender' feature to a binary choice of either 0 or 1. From here, we decided to individualize the different preprocessing we used for each model.
      * For the Logistic Regression Model (which also contains our data exploration), we transformed our modified dataset and normalized the values of our features, and we used ‘Survived’ as our target variable when splitting our dataset into train and test datasets.
      * For the Kmeans Cluster Model, we created a new feature labeled ‘Family_Size’, we dropped the features 'Siblings/Spouses Aboard' and 'Parents/Children Aboard', and we used ‘Pclass’ as our target variable when splitting our dataset into train and test datasets.
      * For the Decision Tree Model, we created a new feature labeled ‘Family_Size’, and we used ‘Pclass’ as our target variable when splitting our dataset into train and test datasets.
      * For the Neural Network Model, we created two new features labeled ‘Family Size’ and ‘Travel Alone’, we transformed our modified dataset and normalized the values of our features, we one hot encoded our ‘Pclass’ feature into three distance features which were ‘1’, ‘2’, and ‘3’ (each number represents the passengers class where 3 is the lower class, 2 is the middle class, and 1 is the upper class), and we used our three distinct Pclass features (‘1’, ‘2’, and ‘3’) as our target variables.
    * Note: Each model does different major preprocessing, but the links for other models are displayed in the subsection of each model.
  * Logistic Regression Model:
    * In the hopes of learning which model could be best for predicting our chosen class within our Titanic dataset, we started of by doing logistic regression, one of the first techniques we learned in this class. In conducting a logistic regression, we used the LogisticRegression method within the sklearn library. To introduce the relevant steps that we took to ensure a successful approach to logistic regresion, we
      * Initated a MinMaxScaler() object and scaled our data
      * Created train and test data for our model by using a 80/20 split respectively
      * Applied predictions using these data and finally checked the model's accuracy.
    * For results of this section, we printed out a classification report, the feature coefficients using a scatterplot and several more scatterplots comparing different features with our target variable ‘Survived’ using their line of best fits.
    * Logistic Regression Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Logistic_Regression_Model_Project_WriteUp.ipynb
  * Kmeans Cluster Model: 
    * As we wanted to expand on what KMeans could do, we decided to create a new feature 'family_size' which was the combination of two already declared features 'Siblings/Spouses Aboard' and 'Parents/Children Aboard.' This feature has also been used in several other models and will be reflected as such. In addition, we decided to drop those said features 'Siblings/Spouses Aboard' and 'Parents/Children Aboard'.
    * KMeans has only one parameter of how many clusters we want, which we set to 3. In analyzing the results of our model, we printed our Mean Squared Error and plotted out several 3D models of how accurately KMeans was able to distinguish each passenger's class based on a varying feature set every time.
    * Kmeans Cluster Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/KMeans_Cluster_Model_Project_WriteUp.ipynb
  * Decision Tree Model: 
    * A sklearn decision tree first requires us to split the train and test data in a 80/20 split respectively and then fit the model using the resulting X_train and y_train data. From here the model takes over to determine what splits are required to accruately determine which features are generally accustomed to one class.
    * Following this, our results include a printed graphic of how our tree looks including the features it split on, and the mean accuracy of the tree.
    * Decision Tree Model Link: https://github.com/CBelleLopez/CSE151A_Proj/blob/main/Decision_Tree_Model_Project_WriteUp.ipynb
  * Neural Network Model: 
    * Before running this model, we also chose to puruse a few additional steps.
      * Added another feature for this model called 'Travel Alone'
      * One-hot encoded the output data and do a train and test split on a 80/20 respectively
    * Built our sequential neural network by having three dense layers.
      1. Our first & input layer has 128 units and uses relu as the activation function
      2. Another layer with 64 units which also has relu as the activation function
      3. Final and output layer has 3 units and has softmax as the activation function.
    * The model then gets compiled using Adam() as the optimizer and checks loss per categorial_crossentropy and uses accuracy as its metric of accuracy. We also used an early_stopping callback mechanic as introduced in HW3 to either end early or continue the training process based on the score. We finally fit our model per our train data using a batch size of 10 running for 100 epochs.
    * The results of this model include a classification report per training and test data, and a confusion matrix.
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
We began this project trying to predict survival rate using logistic regression, but we were told last minute this was too simple so we had to think more abstractly. We were able to come up with k means clustering, a decision tree, and a neural network model to predict the class of a passenger. One thing we could do differently in the future is dive deeper into these models and optimize them further, varying the amount of nodes in the neural network and experimenting with various activation functions. We could also perform cross validation to further validate our results. We could also think of new ways to use the new features we created and draw further conclusions about the interrelationships among features. Overall, this project successfully demonstrated the application of various modeling techniques on the Titanic dataset, and there is always room for improvement in the predictive capabilities of the models. 

## Statement of Collaboration
Andrew Lu: Group Member - 

Bryant Quijada: Group Member -  

Clarabelle Lopez: Group Member -  

Fayaz Shaik: Group Member -  

Jordan Phillips: Group Member -  

Owen Lam: Group Member -  

---

Milestone 2 Update:

We already started on the preprocessing step due to our not being able to properly graph the names feature since those were entered as strings and there is no possible way for us to encode that to numerical values. The sex features were encoded to [0,1] where 0 represented females and 1 represented males. Future preprocessing steps that we could take is dropping fares that cost zero dollars as that would skewer the correlation between fare cost and survival chances. Normalization would have to be implemented for fare and age since the numerical values can increase or decrease drastically which can affect the sensitivity of our algorithm. Nornmalizing them to the same as sex and survival would allow our features to have the same range making it easier to find correlation. 

Milestone 3 Update:

This milestone involved us dealing with data pre-processing. We first tried this out by training a logistic regression model to our data, which involves the general steps of scaling the data and splitting the dataset. We printed out a classification report to see our precision and recall and our model did fairly well with a 79% accuracy. Even printing our scatterplot displayed a check-mark eqsue shape which indicated that the second feature was the least important and growing more important with each subsequent feature. We printed out several scatterplots with their line of best fits, whose discussion and results are included in our notebook. To improve our results, we could maybe have done further normalization and include more datasets to reflect a better prediction on the relations between these features. Our results can be seen within our notebook.

---



<a target="_blank" href="https://colab.research.google.com/github/CBelleLopez/CSE151A_Proj/blob/main/Project_WriteUp.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
