This repository holds code to update the drivers of tree cover loss dataset on Resource Watch.

File structure:
- inputs: plot data, labeled tiles, landsat imagery 
- outputs: predicted tiles
- unetModels: jupyter notebooks for preprocessing inputs and running models
- utils: python helper functions for modeling
- viz: visualizations of predictions

Run order for unetModels:
1. exportTCLOverPlots.ipynb: export tree cover loss over the plot, includes all years of loss
2. rasterizePlots.ipynb: once you have the tree cover loss exported, this notebook rasterizes the labeled samples, and separates them into separate files of the format plot_PLOTID_YEAR.tif
3. exportLandsatOverPlots.ipynb: export landsat imagery over plots by year
4. separateTrainingValidation.ipynb: splits plot ID's into training and validation, splitting by Census Region and Sampled From attributes
5. firstModel.ipynb: running ML models
6. visualize.ipynb: visualize inputs and outputs
