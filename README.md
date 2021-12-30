# Lab - Data Analytics Library

## Overview
Lab is a small library for python data analysis, made with the purpose of easing
the workflow of analysing data, to get a quick overview of the most important aspects of the data at hand.
It contains features such as,

 - Easy acces to different analysis types
 - Fitting data to models and visualise residuals
 - Clean data for missing values and inconvertable values
 - Keeping af log of events in the analysis progress

All made with the purpose of creating an simple and quick workflow for getting a quick overview of the data.

## Methods

## Workflow
The intended workflow using the Lab library is the following:

```
from Lab import *
```
  
This imports all the nescesarry libraries and classes properply.
Afterwards a Lab instance should be created.

```
lab = Lab()
```

With the Lab instance, experiments can be added using the .add_experiment method. Its parameters are explained in the Methods subsection.

```
data = sns.load_dataset('iris')
lab.add_experiment(data, name="Iris data")
```

Here i used the famous iris dataset. Afterwards the appropoiate analysis can be made, fx a standard analysis.

```
lab.analysis(name="Iris data", analysis_type="Standard", save=True)
```

Alternatively, a Bootstrap, Scatter or Frequency analysis could have been made. The analysis saves the folowing pairplot:

![](/Pairplot_Iris_data.png)

When you got an overview of the data, its time to see how your model fits the data.

```
# Define a fit function
def func(x, a, b):
    return a*x + b

# Define errors
sigma = [[0.5]*150]

# Perform fit
popts, perrs = lab.fit(name="Iris data", columns=[['petal_width','sepal_length']], funcs=[func], sigma=sigma, absolute_sigma=True, residuals=True, save=True)
```

The above code, defines a model (function), makes a list of errors for the dependent variable and uses the .fit method to get the optimal parameters
and errors for those parameters. The saved figure looks like this.

![](/Fit_petal_width_sepal_length.png)

The analysis progress might not be linear, so the .update, .clean methods are given, to update and clean data.
Finally the Lab instance keeps a log of your progress, which you can edit with the .add_note method, and
which is accessible through the .log method.
All of the above are illustrated in the lab_notebook_test.ipynb notebook.

## Download Guide

## References
See my other [projects](https://fred465f.github.io/Data_Analytics_Portfolio/)!

## Version
Version 1.0. Updated 30-12-2021.
