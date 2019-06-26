# Tasks

+ **Data exploration**
    + How much does change the T across Barcelona?
    + Are there correlations between days of the same month?
    + Which are the most relevant variables for the mean temperature? Humidity? Wind? Solar radiation?

+ **Data retrieval**
    + Meteocat
        + They have an API
    + Meteoclimatic
    + Corin
    + NASA?

+ **Data cleaning and preparation**
    + Are we going to predict for the first challenge at the grid level?
    + We have only months\*points\*year (12 / 5 / 6) = 360 values for predicting

+ **Models**
    + Input for each model?
    + List of models (given from the dumbest to the "smartest"):
        + Mean of stations
        + Linear regression
        + ...
    + Crossval

# OnGoing

+ Juan
	+ I have begin implementing a data pipeline. My idea is to have a common ground for data pipelining. The file is called utils reader. By the moment it only includes a downloader and 
	a dataframe grouper. Feel free to modify it.
