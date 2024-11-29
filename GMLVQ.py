import pandas as pd
import prototorch as pt

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class preprocess_feature():
    """tiền xử lý dữ liệu và tính toán các đặc trung.
    Parameters
    ------------
    n : int
        kích thước của vecto đầu vào
    m : int
        bước nhảy sau mỗi lần trích xuất vecto đầu vao
    file_path: str
        đường đến file csv chứa dữ liệu
    Returns
    -------
        self : object
    """
    def __init__(self, n, m, file_path):
        self.n = n
        self.m = m
        self.file_path = file_path
    def preprocess(self):
        #tạo vecto đầu vào và nhãn dán tương ứng từ dữ liệu đọc được trong file CSV
        #đọc dữ liệu từ file CSV
        df = pd.read_csv(self.file_path, header=0)
        #tạo list để lưu trữ các vecto
        vectors = []
        #lặp qua dữ liệu để tạo vecto
        for i in range(0,len(df)-self.n+1,self.m):
            #lấy n phần tử từ cột đầu tiên của dataframe
            input_vector = df.iloc[i:i+self.n,0].tolist()
            #lấy n phần tử từ cột thứ hai của dataframe
            label_vector = df.iloc[i:i+self.n,1].tolist()
            if len(np.unique(label_vector)) >1:
                continue

            #tạo một dictionary để lưu trữ vector và nhãn dán tương ứng
            vector_data = {'input_vector': input_vector, 'label': label_vector[0]}
            #thêm dictionary này vào list của các vecto
            vectors.append(vector_data)

        #chuyển list của các vecto thành DataFrame mới
        new_df = pd.DataFrame(vectors)
        return new_df
    def mean(self):
        #tính giá trị trung bình của mỗi vector đầu vào
        mean_value = self.preprocess()['input_vector'].apply(lambda x: np.mean(x))
        # #chuyển mean_value về kiểu nguyên
        # mean_value = mean_value.astype(int)
        return np.array(mean_value)
    def minimum(self):
        #lấy giá trị nhỏ nhất của mỗi vector đầu vào
        min_value = self.preprocess()['input_vector'].apply(lambda x: np.min(x))
        return np.array(min_value)
    def maximum(self):
        #lấy giá trị lớn nhất của mối vector đầu vào
        max_value = self.preprocess()['input_vector'].apply(lambda x: np.max(x))
        return np.array(max_value)
    def standard_deviation(self):
        #tính độ lệch chuẩn của mỗi vector đầu vào
        std_value = self.preprocess()['input_vector'].apply(lambda x: np.std(x))
        return np.array(std_value)
    def slope(self):
        slopes = []
        df = self.preprocess()
        for i in range(len(df)):
            input_vector = df.iloc[i]['input_vector']
            argmax_index = np.argmax(input_vector)
            argmin_index = np.argmin(input_vector)
            if argmax_index != argmin_index:
                slope_value = (np.max(input_vector) - np.min(input_vector)) / (argmax_index - argmin_index)
            else:
                slope_value = 0
            slopes.append(slope_value)
        return np.array(slopes)
    

#tien xu ly
file_path = r"F:\ky_2023_2\do_an_1\do_an_code\file_1.csv"
pf = preprocess_feature(n=200, m=50, file_path=file_path)
#goi cac phuong thuc ra de tao dataframe moi
mean = pf.mean()
minimun = pf.minimum()
maximum = pf.maximum()
standard_deviation = pf.standard_deviation()
slope = pf.slope()
labels = np.array(pf.preprocess()['label'].values)
#tao dataframe moi tu cac dac trung tren
data = {
    'mean':mean,
    'minimum':minimun,
    'maximum':maximum,
    'standard_deviation':standard_deviation,
    'slope':slope,
    'labels':labels,
}     
data_features_label = pd.DataFrame(data)
# print(data_features_label.head())

X, y = data_features_label.iloc[:, 0:-1].values, data_features_label.iloc[:, -1].values
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.3, random_state=0, stratify=y0)
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.fit_transform(X_test)
# X = np.concatenate((X_train_std, X_test_std), axis=0)
# y = np.concatenate((y_train, y_test), axis=0)
# X = stdsc.fit_transform(X)
for i in range(len(X)):
    X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])


class DataWrapper(Dataset):
    def __init__(self, data):
        self.data = data  # list [x, y]

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        data_idx = idx % len(self.data[0])
        x = torch.tensor(self.data[0][data_idx])
        y = torch.tensor(self.data[1][data_idx])

        return x, y
    
def get_loader(args, data):
    
    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y
    
    dataset = DataWrapper(data)
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(5))

    train_loader =  DataLoader(dataset=train_dataset, 
                                drop_last=True,
                                shuffle=True,
                                collate_fn=collate_fn,
                                batch_size=args.batch_size,
                                num_workers=0)
    dev_loader =  DataLoader(dataset=val_dataset, 
                                drop_last=True,
                                shuffle=False,
                                collate_fn=collate_fn,
                                batch_size=args.batch_size,
                                num_workers=0)

    return train_loader, dev_loader


class GMLVQ(torch.nn.Module):
    """
    Implementation of Generalized Matrix Learning Vector Quantization.
    """
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)

        self.components_layer = pt.components.LabeledComponents(
            distribution=[5, 7],  # number of codebooks each label
            components_initializer=pt.initializers.SMCI(data, noise=0.1),
        )

        # Initialize Omega matrix
        self.backbone = pt.transforms.Omega(
            5,
            5,
            pt.initializers.RandomLinearTransformInitializer(),
        )

    def forward(self, data):
        components, label = self.components_layer()
        latent_x = self.backbone(data.unsqueeze(1) - components) ** 2 # (x - w) @ Omega.T
      
        distance = torch.sum(latent_x, dim=-1)

        return distance, label

    def predict(self, data):
        """
        Predict the winning label from the distances to each codebook.
        """
        components, label = self.components_layer()
        distance = torch.sum(self.backbone(data.unsqueeze(1) - components) ** 2, dim=-1)
        winning_label = pt.competitions.wtac(distance, label)
        return winning_label

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_loader, dev_loader = get_loader(args, [X, y])
    model = GMLVQ(train_loader).double()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = pt.losses.GLVQLoss(transfer_fn='identity')

    for epoch in range(1000):
        correct = 0.0
        for x, y in train_loader:
            d, labels = model(x)
            loss = criterion(d, y, labels).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred = model.predict(x)
                correct += (y_pred == y).float().sum(0)
        acc = 100 * correct / (len(train_loader) * args.batch_size)
        print(f"Epoch: {epoch} Accuracy: {acc:05.02f}%")

    code_book, labels = model.components_layer()
    omega = model.backbone


    correct_test = 0.0
    with torch.no_grad():
        for x, y in dev_loader:
            d, labels = model(x)
            y_pred = model.predict(x)
            correct_test += (y_pred == y).float().sum(0)
        acc_test = 100 * correct_test / (len(dev_loader) * args.batch_size)
        print(f"Accuracy test: {acc_test:05.02f}%")

    print(code_book)
    print("/n")
    print(labels)
    print("/n")
    print(omega.weights)