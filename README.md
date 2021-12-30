# Lab - Data Analytics Library

## Overview
Lab is a small library for python data analysis, made with the purpose of easing
the workflow of analysing data, to get a quick overview of the most important aspects of the data at hand.
It contains features such as,

 - Easy acces to different analysis types
 - Fitting data to models and visualise residuals
 - Clean data for missing values and inconvertable values
 - Keeping af log of events in the analysis progress (editable)

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


## Download Guide

## References
See my other [projects](https://fred465f.github.io/Data_Analytics_Portfolio/)!
