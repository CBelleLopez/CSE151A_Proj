# CSE151A_Proj

We already started on the preprocessing step due to our not being able to properly graph the names feature since those were entered as strings and there is no possible way for us to encode that to numerical values. The sex features were encoded to [0,1] where 0 represented females and 1 represented males. Future preprocessing steps that we could take is dropping fares that cost zero dollars as that would skewer the correlation between fare cost and survival chances. Normalization would have to be implemented for fare and age since the numerical values can increase or decrease drastically which can affect the sensitivity of our algorithm. Nornmalizing them to the same as sex and survival would allow our features to have the same range making it easier to find correlation. 

Milestone 3 Update:

THis milestone involves us dealing with our data pre-processing. We first tried this out by training a logistic regression model for our data, which involves the general steps of scaling the data and splitting the dataset. We printed out a classification report to see our precision and recall but our model did fairly well with a 79% accuracy. Even our scatterplot when printed displaying a check-mark eqsue shape which indicated that the second feature was the least important and growing important with the each subsequent feature. We printed out several scatterplots with their line of best fits, whose discussion and their results are included in our notebook. To improve our results into a better udnerstanding, we could maybe have done further normalization and include more datasets to reflect a bettter prediction on the relations between these features. Our results can be seen within our notebook.



<a target="_blank" href="https://colab.research.google.com/github/CBelleLopez/CSE151A_Proj/blob/main/Project_WriteUp.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
