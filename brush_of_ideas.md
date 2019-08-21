# Situation

Hi guys, basically back on the track.

If you exeute utils reader either from the cli or calling the function
`prepare_data`, the dataframes fro X and y are obtained. 

Maybe I shouldn't do this split of X and y. I will corect it soon.

From this X and y you can test models if desired. I will be uploading more soon.


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

# Ongoing

+ Juan
	+ Full pipeline of features + models wcikit way
	
+ Biel
	+ Geographic features for each cell: green level, altitude, ...

+ NEXT STEPS:
	+ Include features from hourly data
	+ Include features from non official
	

# Ideas
	
+ Map as an image
	+ Maybe like sparse matrices?
	+ Deconv Networks
	+ Taking polygons with places without information: when computing loss only taking 
	+ https://www.mdpi.com/2072-4292/11/5/597/pdf
+ ANN: multilayer perceptron
	+ https://pdfs.semanticscholar.org/53e4/884eafff5082d26495125dc654c4efb4915a.pdf
	+ http://www.diva-portal.org/smash/get/diva2:1204647/FULLTEXT02
+ RNN for consecutive days
