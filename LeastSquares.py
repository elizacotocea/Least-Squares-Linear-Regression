# learning step: init and train a linear regression model y = f(x) = w0 + w1 * x
# Prediction step: used the trained model to estimate the output for a new input



# # using sklearn
# from sklearn import linear_model

# # training data preparation (the sklearn linear model requires as input training data as noSamples x noFeatures array; in the current case, the input must be a matrix of len(trainInputs) lineas and one columns (a single feature is used in this problem))
# xx = [[el] for el in trainInputs]

# # model initialisation
# regressor = linear_model.LinearRegression()
# # training the model by using the training inputs and known training outputs
# regressor.fit(xx, trainOutputs)
# # save the model parameters
# w0, w1 = regressor.intercept_, regressor.coef_[0]
# print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')

