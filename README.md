# Deploy-Diabetes-model-with-AzureML

# **Table of Contents**

* Overview
* Workspace and Architecture 
* Dataset
* Project Steps
* AutoML Experiment
* * AutoMl Results
* HyperDrive Hyperparameter Tuning
* * HyperDrive Tuning Results
* Best Training Method
* Model Development
* Deployed Endpoint Testing
* Future Improvements
* Screencast Video
* References

## **Overview**

This project is final capstone of the Udacity Azure ML Engineering Nanodegree. In this project, I built and optimized Azure to configure a cloud-based machine learning production model. I deployed and tested the best model through an endpoint of the deployed model. 

## **Workspace and Architecture** 

Workspace forms the top level of resource of Azure Machine Learning.  The workspace is use to manage data, compute resources, code, models, and other artifacts related to machine learning workloads.![workspace](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/ws.png)

The images below indicate the architectural diagram and the overall workflow of the project:

![architecture](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/arch.png)

![workflow](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/wf.png)

## **Dataset** 

The dataset for this project was obtained from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database?select=diabetes.csv), and was originally from the National Institute of Diabetes and Digestive and Kidney Diseases contain medical records of patients to predict the propensity of a patient having diabetes. Diabetes is a metabolic disease that causes high blood sugar over a prolong period of time. Some of the common symptoms include frequent urination, increased thirst and appetite (Wikipedia).

The following medical features were used to predict whether a patient will be diabetic or not:
1.    Pregnancies - Number of times pregnant
2.    Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3.    Blood pressure - Diastolic blood pressure (mm Hg)
4.    Skin thickness - Triceps skin fold thickness (mm)
5.    Insulin – 2 hour serum insulin (muU/ml)
6.    BMI – body mass index(weight in kg)/(height in m)^2
7.    Diabetes pedigree – diabetes pedigree function
8.    Age – age of patient (years) and 
9.    Outcome – class variable (0 or 1) 268 of 768 are 1 and the others are 0.

The target (Outcome) was to do a binary classification using a logistic regression model indicating whether an individual will be diagnosed to be diabetic (1) or not (0). The dataset was saved in the project folder as a csv file, diabetes.csv.

## **Project Steps**

The workflow essentially involves using Microsoft Azure ML SDK to build a model by applying two different training methods. The first method involved using automated machine learning (automl) to train and find the best model that fit the diabetes dataset. The second method involved using HyperDrive to optimize the hyperparameters of the logistic regression model to find the best parameters that fit the same diabetes dataset. The best model was registered and then deployed as a web service with the sdk. The deployed model was tested with a REST API and sdk to predict the outcome.


## **AutoML Experiment**

A new compute instance was created to run the included jupyter notebooks. The diabetes dataset was uploaded from my GitHub repository in Tabular form into Azure Machine Learning Datastore so that it could be used during model training. An experiment was created using automated machine learning (ML) by configuring a compute cluster, and using that cluster to run the experiment. The automl process explored a wide range of algorithms  and parameter sets in parallel that went through a series of iterations that eventually converged to produce models with associated training score. The model with the highest training score based on the criterion defined in the experiment was selected as the best model to fit the dataset. 

The AutoML settings and configuration is shown in the image below:

![automl-settings](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r1.png)

The experiment timeout was set to control the use of the cloud resources. A maximum of 4 iterations were selected to simultaneously run the experiment to maximize usage. The primary metric for the binary (0,1) classification task performed to predict the target column Output (0,1) was Accuracy. The Featurization was set to “auto” in order to automatically scale and normalize the dataset. A cross validation splits of 5 was set to reduce model training overfitting and variance.

The screenshots below show the run details and the best model in terms of accuracy:

![run details ](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r2.png)

![dataset](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r3.png)

### **AutoML Results**
The best performing model selected was VotingEnsemble with an accuracy of 78.6 %. 

The models and it’s weights that the VotingEnsemble used is listed below:

'ensembled_algorithms': "['XGBoostClassifier', 'GradientBoosting', 'LogisticRegression', 'GradientBoosting', 'XGBoostClassifier', 'RandomForest', 'XGBoostClassifier', 'RandomForest', 'ExtremeRandomTrees', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier']" 

'ensemble_weights': '[0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.14285714285714285, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.14285714285714285]'

The screenshots below indicate detatils of the best selected model listed in Azure ML Studio:

![best model details ](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r4.png)

![model metrics](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r5.png)

**Model Best run ID**

![best run id](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r6.png)

**Best AutoML Model Registered**

![best automl model registered](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r7.png)

## **HyperDrive Hyperparameter Tuning**

The HyperDrive configuration is shown in the image below:

![hyperdrive-settings](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r8.png)

The logistic regression hyperparameters that were optimized were the inverse regularization (C) float value and the maximum iterations (max_iter) integer value. The numerical value “C ” defines the inverse regularization strength that helps to prevent model overfitting. The smaller the C value the stronger the regularization. 
The 'max_iter' defines the maximum number of iterations taken for the solvers to converge with a default value of 100. 

The **Random sampling** supports discrete and continuous hyper parameterization. It also supports the early termination of low-performance runs. 
The Random sampling parameter **-C** was provided choice values of 0.2, 0, 5 or 1.0. The second Random sampling parameter  **--max_iter**  was provided as choice values of 10, 20 or 30. The early termination policy used was **BanditPolicy** based on a slack factor and evaluation interval. The Bandit terminates runs when the primary metric is not within the specified slack factor compared to the best performing run.

The ScriptRunConfig method was used for configuring the HyperDrive  training job in Azure Machine Learning via the SDK. A ScriptRunConfig packages together the configuration information needed to submit a run in Azure ML, including the script, compute target, environment, and any distributed job-specific configs.
In using the ScriptRunConfig, all environment-related configurations have to be encapsulated in the Environment object that gets passed into the environment parameter of the ScriptRunConfig constructor. 
In order to configure this training job, I provided a custom Environment through a yml file called conda_dependencies.yml, which contains all the dependencies required for my training script (train.py). 

** HyperDrive Run Details**

![run details1](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r9.png)
![run details2](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r10.png)
![run details2](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r11.png)

### **HyperDrive Tuning Results**

The best Accuracy for the HyperDrive model was 72.7 %  with the following best selected hyperparameters :

* Regularization Strength ( --C): 0.2 and 
* Max iterations (--max_iter): 30

The images below show the hyperdrive model training results:

![training accuracies](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r12.png)
![best model](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r13.png)


**Best HyperDrive Model Registered**

![best automl model registered](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r14.png)


### **Best Training Method**

* The accuracy of AutoML (0.786) is higher than the accuracy of HyperDrive (0.727) by **0.059 **.
* The HyperDrive used a simple logistic regression architecture which is easy to configure while AutoML used more complex parameter configuration.
* AutoML has the advantage of setting and comparing several models in parallel with different hyperparameter settings while HyperDrive will train one model at the time.
* HyperDrive model required the setting up of the provided train.py script where the logistic regression was calculated over the weighted sum of the input passed via the sigmoid activation function. AutoML did not use the train.py script.
* HyperDrive is resource intensive and can be manual while AutoML is efficient, when setting up multiple estimators.
* In general, AutoML appeared to be better than HyperDrive as it allows the selection of the best model among several models running in parallel and allows the Data Scientist to focus more on business problems.

## **Model Development**

Since the autoML model had the highest accuracy, it was further deployed and test through the sdk. 
The best autoML model was deployed by preparing an inference configuration,  which describes how to set up the web-service containing the model. An entry script (score.py) was prepared to receive data submitted to the deployed web service and passed it to the model. The response from the model is returned to the client. The entry script also served as an input to InferenceConfig. The Azure container instance (ACI) was chosen as the compute target to host the model and this influences the cost and the availability of the deployed endpoint. The web services eventually took the model, loaded it in an environment and run it on one of the supported deployment targets.

The following screenshots show the status of the automl deployment described as "healthy":

![best automl model registered](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r14.png)

![sdk deployment](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r15.png)

![ml studio deployment status](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r16.png)

## **Deployed Endpoint Testing**

The deployed model was tested in two ways:
* A REST endpoint file (endpoint.py) was supplied with a scoring url generated from the deployment, and was used to call the service by passing two data points as json predictions as shown below.

* Two random sample data points were passed to compared the actual values and predicted values and the results are shown below.

![testing](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r17.png)

**Logs of the Web Service**
The screenshot below depicts the logs generated from the deployed web service:

![logs](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r18.png)

**Service Deletion**

![service deletion](https://github.com/eaamankwah/Deploy-Diabetes-model-with-AzureML/blob/main/screenshots/r19.png)

## **Future Improvements**

In future, the following areas could be improved:

* More data should be collected to improve the accuracy of the model as the algorithms can learn more from more dataset.

* Different target metrics could be pursued to broaden the range of model evaluation choices. The range of range of the hyperparameters could be increased to see if further improvements could be achieved.
* Automate efficient hyperparameter tuning by using Azure Machine Learning HyperDrive package could be used for comparison purposes . Different combination of  hyperparameter values  --C and --Max_iter could be tried. The C value could be selected by using the Uniform or Choice functions to define the search space. The parameter search space could be defined as discrete or continuous, and a sampling method over the search space as grid, or Bayesian. New HyperDrive runs with difference estimators including the best performing estimator VotingEnsemble could be tried to improve the accuracy score. 

* It is also recommended to increase cross validation in search of a better model accuracy.

* The Deep Learning option in autoML could be enabled for the binary classification task to ensure that the text data are classified. Deep learning models can improve the accuracy but may be hard to interpret.

* Since the dataset is not “big” and not balanced, tuning the hyperparameter to a full completion may improve the accuracy score and using an AUC_weighted metric or balancing the dataset may improve the accuracy results.

* Although it is recommended to decrease the exit criterion of the autoML model in order to save compute resources, this criterion could be increased to identify the model with the best performance.

## **Screencast Video Link**

The link below demonstrates the main processes undertaken in this project

https://drive.google.com/file/d/1QQtM7B1swHT6BxHUmWCCXQqxqKPl0ojQ/view?usp=sharing

## **References** 

* [ScriptRunConfig](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-migrate-from-estimators-to-scriptrunconfig)

* [Azure Machine Learning Pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines)

* [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)

* [Udacity Q & A Platform](https://knowledge.udacity.com/?nanodegree=nd00333&page=1&project=755&rubric=2893)
