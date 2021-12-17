This repository holds code to update the drivers of tree cover loss dataset on Global Forest Watch Watch.

File structure:
- unetModels: jupyter notebooks for preprocessing inputs and running models
- gcloudModels: google cloud ai platform 

Run order for unetModels:
1. exportTCLOverPlots.ipynb: export tree cover loss over the plot, includes all years of loss
2. rasterizePlots.ipynb: once you have the tree cover loss exported, this notebook rasterizes the labeled samples, and separates them into separate files of the format plot_PLOTID_YEAR.tif
3. exportLandsatOverPlots.ipynb: export landsat imagery over plots by year
4. separateTrainingValidation.ipynb: splits plot ID's into training and validation, splitting by Census Region and Sampled From attributes
5. firstModel.ipynb: running ML models
6. visualize.ipynb: visualize inputs and outputs
