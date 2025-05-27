# NN_methods.py

'''This code is based on a course in neural networks for vision tasks in Microsoft Learn'''

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin #  s.th. we can use get_params, set_params, score and more from scikit learn
from sklearn.pipeline import Pipeline
import torch.optim as optim
import numpy as np

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_layer_sizes=(100,), output_size=1,learning_rate=1e-3, patience=20, weight_decay=1e-4,
                optimizer_class=optim.Adam, activation=nn.ReLU, num_epochs=25, loss_func_class=nn.BCELoss):
        
        # self.flatten = nn.flatten()       # No need to flatten data when it is only one channel matrix
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.activation = activation
        self.num_epochs = num_epochs
        self.loss_func_class = loss_func_class
        self.patience = patience
        self.scaler = RobustScaler()
        self.weight_decay = weight_decay    # Apply Lasso to the optimizer used
        self.model = None
        
    def _initialise_model(self):
        
        self.model = self._assemble_model()
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = self.loss_func_class()
    
    def _assemble_model(self):
        
        layers = []
        input_size = self.input_size
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.activation())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def _train_model(self, train_loader):
        
        size = len(train_loader.dataset)
        best_loss = float('inf')
        last_improvement = 0
        for epoch in range(self.num_epochs):
            for batch, (X, y) in enumerate(train_loader):
                # Compute prediction and loss
                outputs = self.model(X)
                loss = self.loss_func(outputs, y)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Trying to make it print how it goes...
                # if batch % 10 == 0:
                # loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    
        # if current loss is better than the best loss
                if loss < best_loss:
                    best_loss = loss
                    last_improvement = epoch
            if epoch - last_improvement > self.patience:
                print("Early stopping!")
                break
    
    
    def fit(self, X, y):
        
        self._initialise_model()
        X = self.scaler.fit_transform(X)    # Normalize the data for a canonical model
        
        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        tensor_training = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(tensor_training, batch_size=32, shuffle=True)
        self._train_model(train_loader)
    
    def predict(self, X):
        
        self.model.eval()
        X = self.scaler.transform(X)    # Normalize test data
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            return (outputs > 0.5).numpy().astype(int)


def NN_method(X, y, n_splits=3, repeatedCV=True, randomize=100, verbosity=3, test_size=0.3):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    input_size = X_train.shape[1]
    
    params_grid = {
    # 'feature_selection__k': [6, 8, 10, 11],
    'model__learning_rate': [1e-5, 0.5e-4, 1e-3#, 5e-3, 1e-2, 5e-2
                             ],
    'model__optimizer_class': [#optim.AdamW,
                               optim.Adam#, optim.SGD
                               ],
    'model__activation': [nn.ReLU,# nn.Tanh, nn.LeakyReLU,
                          nn.Sigmoid
                          ],
    'model__num_epochs': [25, 50, 
                          100, 150, 250
                          ],
    'model__loss_func_class': [nn.BCELoss#, nn.BCEWithLogitsLoss
                               ],
    'model__patience': [10,# 20, 
                        30, 50, 100
                        ],
    'model__hidden_layer_sizes': [(50,), (100,), (32, 64), (150, 50)#, (500,)
                                  #(250, 200, 75),
                                  #(100, 100, 100)#, (50, 100, 100, 100, 50)#, (25, 50, 100, 100, 150, 100, 100, 75, 50)
                                  ],
    'model__weight_decay': [1e-6, 1e-5, 1e-4, 5e-3, 1e-3
                            ]
    }
    neural_net = NeuralNetwork(input_size=input_size)
    
    pipe = Pipeline([
        # ('feature_selection', SelectKBest(score_func=f_classif)),
        ('model', neural_net)
    ])
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if randomize > 0:
        cv_nn = RandomizedSearchCV(pipe, param_distributions=params_grid, n_iter=randomize, cv=skf, verbose=verbosity, random_state=42, n_jobs=-1)
    else:
        cv_nn = GridSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)

    cv_nn.fit(X_train, y_train)
    
    nn_params = cv_nn.best_params_
    nn_accuracy = cv_nn.best_score_
    nn_results = cv_nn.cv_results_
    
    y_pred = cv_nn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    final_model = cv_nn.best_estimator_
    final_model.fit(X, y)
    
    print(f"Neural network accuracy on training data: {nn_accuracy}\n"
        f"Best neural network parameters: {nn_params}")
    print(f"Neural network accuracy on test data: {test_accuracy}")
    
    return final_model, nn_results