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

## Methods and Properties

### Methods
**Lab.log(self):**
 - Prints current log note containing analysis made, data added and the
time at which the change was made

**Lab.add_note(self, note=None, line=None):**
 - Adds a note to the log on the line given. Line should be an integer
and note should be a string.

**Lab.add_experiment(self, data, name=None):**
 - Adds a experiment to the lab instance with the given name
containing the given data. Data should be a pandas DataFrame
and name should be a string. If name .

**Lab.update(self, data, name):**
 - Updates the data of a experiment with the given name
containing the given data. Data should be a pandas DataFrame
and name should be a string.

**Lab.clean(self, name, columns, pattern=None, filename=None, regex=False):**
 - This method does two things. Firstly, if provided, the value of the variable pattern is removed
from all values of every column. The regex bool determines if pattern will be interpreted as a regex pattern.
Secondly it looks through every row of data from the experiment named after the name variable, and
looks for values that it were not able to convert to a float, saves the index for that row,
and ultimately removes all rows that couldnt be converted to a float, and then converts every column to have float
values only. If filename is given, the data is uploaded. Filename should be a string and end on .csv.
Columns should be a list, name and pattern should be a string and regex should be a bool.

**Lab.analysis(self, name, columns=None, analysis_type="Standard", save=False, tolerance=0.5):**
 - This method performes one of the following analysis, Standard, Bootstrap, Scatter or Frequency
The different analysis, are showed in use in the lab_notebook_test.ipynb notebook.
If save is True, the figure will be saved in the current directory.
Columns should be a list of lists of columns from the DataFrame each of length two,
name should be a string, save a bool and tolerance a float.

**Lab.fit(self, name, columns, funcs, guesses=None, sigma=None, absolute_sigma=False, method=None, residuals=False, save=False, alpha=1):**
 - This method fits a given function to data in columns and uses guesses as a guess for
the optimal parameters. Sigma is the error for the dependent variable, which is taken
in account if absolute_sigma is True. Method is the method for finding the optimal parameters,
by default Least squares method. If residuals is true, then they are plottet. Alpha determines the transparency.
If save is True, the figure will be saved in the current directory.
The fit method are showed in use in the lab_notebook_test.ipynb notebook.
Columns should be a list of lists of columns from the DataFrame each of length two,
name should be a string, save a bool, funcs, sigma and guesses a list, absolute_sigma a bool
method a string, residuals a bool and alpha a float.

### Property
**Lab.experiments:**
 - Returns the experiments in a dictionary.

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
