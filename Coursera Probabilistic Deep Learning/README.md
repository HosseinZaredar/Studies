# Coursera Probabilistic Deep Learning

Coursera `Probabilistic Deep Learning with TensorFlow 2` course, taught by Kevin Webster [[Link](https://www.coursera.org/learn/probabilistic-deep-learning-with-tensorflow2)].

## Course Homeworks
- [x] HW1: Naive Bayes and Logistic Regression
- [x] HW2: Uncertainty Estimation using Bayesian Neural Networks
- [x] HW3: Normalising Flows (RealNVP)
- [x] HW4: VAEs (on CelebA Dataset)

## Uncertainty Quantification in Regression Models
When should we trust the predictions of our model? This is the main question we're addressing here.

Usually, regression models output a single or a set of target values. In this way, there's no way for the model to tell us how certain it is about the output(s) it has predicted, and hence, no way for us to know if we can trust the outputs.

In Uncertainty Quantification, we're interested in having a numerical and accurate measure of model's uncertainty about its predictions.

Where does uncertainty come from? It can be inherent in the data (called **Aleatoric Uncertainty**) or it can arise from a noisy model (called **Epistemic Uncertainty**). You can read more about this topic in [our presentation](https://prezi.com/view/aluKQJx8qZGj6hcOtBRk/) or in the resources mentioned below.

Different methods have been proposed in the recent years to deal with uncertainty. In this project, 3 methods are implemented:
1. Weight Uncertainty Using Bayesian Deep Learning
2. Model Ensembling
3. Monte Carlo Dropout

### Results

The variance in the plots shows the uncertainty of the model:

Bayesian Networks            |  Model Ensembling          |  Monte Carlo Dropout
:-------------------------:|:-------------------------:|:-------------------------:
![Bayesian](https://github.com/HosseinZaredar/Studies/blob/main/Coursera%20Probabilistic%20Deep%20Learning/images/bayesian.png?raw=true)  |  ![Ensembling](https://github.com/HosseinZaredar/Studies/blob/main/Coursera%20Probabilistic%20Deep%20Learning/images/ensemble.png?raw=true) | ![Monte Carlo](https://github.com/HosseinZaredar/Studies/blob/main/Coursera%20Probabilistic%20Deep%20Learning/images/monte.png?raw=true)


All the code can be found [here](https://github.com/HosseinZaredar/Studies/blob/main/Coursera%20Probabilistic%20Deep%20Learning/Uncertainty-in-Regression.ipynb).

## Other References
- [Why Uncertainty Matters in Deep Learning and How to Estimate It](https://everyhue.me/posts/why-uncertainty-matters/)
- [Yarin Gal - Bayesian Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/bdl101/)
