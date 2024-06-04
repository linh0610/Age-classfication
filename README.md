# Age-classfication
 - This project started when I was interested into see how a model that have been trained to an acceptable validation accuracy would classifiy me and my friends at what age
## Dataset
 - This model is trained on the UTK dataset: https://www.kaggle.com/datasets/jangedoo/utkface-new/code
### Turning point 1:
 - When I first start the project, I opted to train with the number class to be 104 since I wanted to predict absolute age, using the model ResNet50 without any additional layers,
I also did cross validation with different nunber of epochs but kept the same lr and wd (well that was stupid of me, I just started learning machine learning after all), I achieved an average accuracy of around 25% pertentage for each fold (k=5).
 - The model was severely underfitting since without any addtional layers the complexity is too low for 104 classes, at this point of time I thought that maybe I can do hyperparameter tuning with different lr and wd using SGD optimizer
### Turning point 2:
 - This time I experiment with different lr and weight decay, with the same model ResNet50 and it accuracy improved, but not much, it only went up to 40% after 4 days of training with different parameter.
 - I also thought of adding additional layers or even combining models but it increased training time significantly when before it was 30 minutes an epoch.
 - I thought maybe I should opted for predicting age range where the changes in facial featrures are not significant, so I went back to prepare the dataset started with hyperparameter tuning with lr = [0.001, 0.005, 0.0005, 0.00025] and wd = [0.001, 1e-4, 1e-5, 0.003], I also change the model from ResNet50 to EfficientNet_B0 instead as I wanted to see the performance of another model. 
 - Here are the age range that I settled with:
folder_names = [
 '1-2',
 '11-14',
 '15-17',
 '18-22',
 '23-27',
 '28-33',
 '3-5',
 '34-40',
 '41-50',
 '51-60',
 '6-10',
 '61-70',
 '71-80',
 'over80']
