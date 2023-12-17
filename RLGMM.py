import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class RLGMM(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) > 1:

            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index
            print(self.feature_df.shape)


            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index=[]
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i] == 0:
                        ex_index.append(i)
            select_index=[]
            for i in range(len(self.feature_df.values[0])):
                if i not in ex_index:
                    select_index.append(i)

            select_index=list(set(select_index))
            sel_feature = self.feature_df.values.T[select_index].T
            columns = self.feature_df.columns[select_index]
            self.feature_df = pd.DataFrame(sel_feature, columns=columns)

            class Autoencoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, encoding_dim):
                    super(Autoencoder, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Linear(hidden_dim, encoding_dim),
                        nn.ReLU(True)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Linear(hidden_dim, input_dim),
                        nn.Sigmoid() 
                    )
                def forward(self, x):
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x


            X_tensor = torch.tensor(self.feature_df.values).float()
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

            input_dim = X_tensor.size(1)
            hidden_dim = 256 
            encoding_dim = 128 

            model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            num_epochs = 100
            for epoch in range(num_epochs):
                for data in dataloader:
                    inputs, = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

            with torch.no_grad():
                encoded_samples = model.encoder(X_tensor).numpy()

            enhanced_df = pd.DataFrame(encoded_samples)


            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

            pass_feature = np.array(self.feature_df[equal_zero_index])
            fail_feature = np.array(self.feature_df[equal_one_index])
            print("length of fail tests",len(fail_feature))
            centr_cc = fail_feature[0]
            dist = []
            for i in range(len(pass_feature)):
                dist.append(np.sqrt(np.sum(np.square(pass_feature[i] - centr_cc))))
            max_dis = max(dist)
            index = dist.index(max_dis)

            initial_center_0 = pass_feature[index]  
            initial_center_1 = fail_feature[0]

            X = pass_feature
            passDF=pd.DataFrame(pass_feature)

            class GMM:
                def __init__(self, initial_means):
                    self.means = np.array(initial_means)
                    self.covariance = None
                    self.weights = np.array([0.5, 0.5])  # Assuming equal weights for both components
    
                def _e_step(self, X):
      
                    likelihoods = np.array([self._pdf(X, self.means[k], self.covariance) for k in range(2)]).T
        
                    weighted_likelihoods = likelihoods * self.weights
                    sum_likelihoods = np.sum(weighted_likelihoods, axis=1)[:, np.newaxis]
                    responsibilities = weighted_likelihoods / sum_likelihoods
        
                return responsibilities

                def _m_step(self, X, responsibilities):
                    Nk = responsibilities.sum(axis=0)
                    self.means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        
                    weighted_squares = np.zeros((2, X.shape[1], X.shape[1]))
                    for k in range(2):
                        diff = X - self.means[k]
                        weighted_squares[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]).sum(axis=0)
        
                    self.covariance = np.sum(weighted_squares, axis=0) / Nk.sum()

                def _pdf(self, X, mean, covariance):
                    n = X.shape[1]
                    diff = X - mean
                    return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1)) / (np.sqrt((2 * np.pi)**n * np.linalg.det(covariance)))

                def fit(self, X, n_iters=100):
                    self.covariance = np.cov(X, rowvar=False)
        
                    for _ in range(n_iters):
                        responsibilities = self._e_step(X)
                        self._m_step(X, responsibilities)

                def predict(self, X):
                    responsibilities = self._e_step(X)
                    return np.argmax(responsibilities, axis=1)
            initial_means = np.stack([initial_center_0, initial_center_1])

            gmm = GMM(initial_means)
            gmm.fit(X)
            pred_classes = gmm.predict(X)
            passDF['Class'] = pred_classes
            class_0_df = passDF[passDF['Class'] == 0].drop(columns=['Class'])
            class_1_df = passDF[passDF['Class'] == 1].drop(columns=['Class'])
            print("Class 0 samples:")
            print(class_0_df.shape)
            print("\nClass 1 samples:")
            print(class_1_df.shape)


            features_np = np.array(fail_feature)
            #compose_tmp = np.vstack((features_np, Tcc1))
            Ttp = np.array(class_0_df)
            compose_feature = np.vstack((features_np, Ttp))

            fnum=len(features_np)
            pnum=len(Ttp)
            flabel = np.ones(fnum).reshape((-1, 1))
            plabel = np.zeros(pnum).reshape((-1, 1))
            compose_label = np.vstack((flabel, plabel))
            print(self.feature_df.shape)
            self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
            self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)
            merged_df = pd.concat([self.feature_df, self.label_df], axis=1)
            shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
            self.feature_df = shuffled_df.iloc[:, :-1]  
            print(self.feature_df.shape)
            self.label_df = shuffled_df.iloc[:, -1] 









