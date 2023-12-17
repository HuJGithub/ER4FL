import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class GaussianMixture_CC(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) > 1:

            '''equal_zero_index = (self.label_df != 1).values
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
            #print(self.feature_df.shape)'''

            '''class Autoencoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, encoding_dim):
                    super(Autoencoder, self).__init__()
                    # 编码器部分
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Linear(hidden_dim, encoding_dim),
                        nn.ReLU(True)
                    )
                    # 解码器部分
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, hidden_dim),
                        nn.ReLU(True),
                        nn.Linear(hidden_dim, input_dim),
                        nn.Sigmoid()  # 如果数据在 [0,1] 范围内则用 Sigmoid，否则可以是 Tanh 或其他函数
                    )

                def forward(self, x):
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x

            # 假设 df 是你的 Pandas DataFrame，每行都是一个样本
            # df = ...

            # 准备数据
            X_tensor = torch.tensor(self.feature_df.values).float()
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 可以调整 batch_size 的大小

            input_dim = X_tensor.size(1)
            hidden_dim = 256  # 编码器和解码器中间层的尺寸
            encoding_dim = 128  # 编码后的尺寸

            # 实例化自编码器模型
            model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, encoding_dim=encoding_dim)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # 训练模型
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

            # 使用训练好的模型计算所有增强后的样本
            with torch.no_grad():
                encoded_samples = model.encoder(X_tensor).numpy()

            # 将增强后的样本转换回 Pandas DataFrame
            enhanced_df = pd.DataFrame(encoded_samples)'''


            # 使用 K-means 算法进行聚类
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

            #centr_tp = pass_feature[index]

            # 给定两个初始的分类中心
            initial_center_0 = pass_feature[index]  # 初始中心对应正类
            initial_center_1 = fail_feature[0]  # 初始中心对应负类

            # 转换DataFrame为numpy数组
            X = pass_feature
            passDF=pd.DataFrame(pass_feature)
            # 将初始中心转换成需要的形状 (n_components, n_features)
            initial_means = np.stack([initial_center_0, initial_center_1])

            # 初始化高斯混合模型，设置两个组件，并且提供初始均值
            gmm = GaussianMixture(n_components=2, means_init=initial_means, random_state=0)

            # 拟合模型
            gmm.fit(X)

            # 预测样本属于哪个组件(类)
            pred_classes = gmm.predict(X)

            # 将预测的类别添加到原始DataFrame作为新列
            passDF['Class'] = pred_classes

            # 分开两类样本
            class_0_df = passDF[passDF['Class'] == 0].drop(columns=['Class'])
            class_1_df = passDF[passDF['Class'] == 1].drop(columns=['Class'])
            # 输出为Pandas DataFrame格式的两类样本
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
            self.feature_df = shuffled_df.iloc[:, :-1]  # 排除最后一列以外的所有列作为数据部分
            print(self.feature_df.shape)
            self.label_df = shuffled_df.iloc[:, -1]  # 最后一列作为标签部分









