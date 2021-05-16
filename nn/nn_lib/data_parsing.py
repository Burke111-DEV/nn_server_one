import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataParser():
    def __init__(self, filename=""):
        self.csv = pd.read_csv(filename)
        self.csv_cols = list(self.csv)[0:5]
    
    def format_csv(self):
        self.df = self.csv[self.csv_cols[1:5]].astype(np.float32)

    def even_split(self, length=-1, months=-6):
        self.df_split = []
        csv_len = len(self.df)
        
        if length > 0 and months > 0:
            print("Provide only a length or a month, not both!")
            return
        elif length > 0:
            n = int(csv_len/(length))
        elif months > 0:
            n = int(csv_len/(months*30))

        self.df_split = [np.array(self.df)[i:i + n] for i in range(0, len(np.array(self.df)), n)]
        return self.df_split

    def overlapping_split(self, length=-1, months=-6):
        self.df_split = []
        days_per_month = 30

        if length > -1:
            print("Length Split Not Available.")
            return

        length = len(self.df)
        step = int((np.abs(months)*days_per_month)/2)
        for i in reversed(range(0, length, step)):
            if (i-(2*step)>0):
                self.df_split.insert(0, self.df[(i-(2*step)) : i])
        # for i in range(0, length, step):
        #     self.df_split.append(self.df[i: i+2*step])
        return self.df_split

    def scale(self):
        if hasattr(self, 'df_split'):
            self.df_scaled = []
            scaler = {}
            for i in range(len(self.df_split)):
                scaler = StandardScaler()
                scaler = scaler.fit(self.df_split[i])
                self.df_scaled.append(scaler.transform(self.df_split[i]))
            self.scaler = scaler
        else:
            self.scaler = StandardScaler().fit(self.df)
            self.df_scaled = self.scaler.transform(self.df)
        return self.df_scaled

    def generate_training_sets(self, past_displacement=1, future_displacement=1):
        # trainX = training set, trainY = true result
        self.trainX, trainX_df, self.trainY, trainY_df = [], [], [], [] 
        
        if hasattr(self, 'df_split'):
            for df in self.df_scaled:
                for i in range(past_displacement, len(df) - (future_displacement)+1):
                    trainX_df.append( df[i - past_displacement:i, 0:df.shape[1]] )
                    trainY_df.append( df[i + future_displacement - 1:i + future_displacement, 0])
                self.trainX.append(np.array(trainX_df, dtype=object))
                self.trainY.append(np.array(trainY_df, dtype=object))
                trainX_df, trainY_df = [], []
            return np.array(self.trainX, dtype=object), np.array(self.trainY, dtype=object)
        else:
            for i in range(past_displacement, len(self.df_scaled) - (future_displacement)+1):
                trainX_df.append( self.df[i - past_displacement:i, 0:self.df.shape[1]] )
                trainY_df.append( self.df[i + future_displacement - 1:i + future_displacement, 0])
            self.trainX, self.trainY = np.array(trainX_df), np.array(trainY_df)
            return self.trainX, self.trainY

    def generate_training_sets_auto(self, even_split=False, overlap_split=False, split_length=-1, split_months=-6, past_displacement=-14, future_displacement=1):
        self.format_csv()
        if even_split:
            self.even_split(length=split_length, months=split_months)
        if overlap_split:
            self.overlapping_split(length=split_length, months=split_months)
        self.scale()
        return self.generate_training_sets(past_displacement=past_displacement, 
                                                 future_displacement=future_displacement)
