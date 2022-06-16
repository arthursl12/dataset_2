import os
import pandas as pd

class DatasetProcessing:
    def download_data(self):
        os.system('git clone https://github.com/arthursl12/dataset_2')
        os.system('mv /content/dataset_2/CMaps /content/CMaps')
        os.system('rm -rf dataset_2')

    def column_names(self):
        index_cols = ['unit_number', 'time']
        settings_cols = ['op_1', 'op_2', 'op_3']
        sensors_cols = [f's_{i}'for i in range(21)]
        cols = index_cols + settings_cols + sensors_cols
        return index_cols, settings_cols, sensors_cols, cols

    def read_dataset(self,scenario=1):
        folder='/content/CMaps/'
        _,_,_,cols = self.column_names()
        train = pd.read_csv(
            (folder+f'train_FD00{scenario}.txt'), 
            sep='\s+', header=None, names=cols)
        test = pd.read_csv(
            (folder+f'test_FD00{scenario}.txt'), 
            sep='\s+', header=None, names=cols)
        y_test = pd.read_csv(
            (folder+f'RUL_FD00{scenario}.txt'), 
            sep='\s+', header=None, names=['RUL'])
        return train, test, y_test

    def transform_test(self,test):
        # Take only the last line for each test set
        # Because it is the only annotated line
        test_last = test.groupby('unit_number').last().reset_index()

        # Dropping unnecessary columns
        idx_c, set_c, _, _ = self.column_names()
        test_last = test_last.drop(idx_c+set_c, axis=1)
        return test_last

    # RUL: how many cycles remain ? 
    #   Take the last (maximun) cycle registered ('time' column) for that sample
    #   Subtract from current cycle number
    #   i.e. assume linear RUL decay
    def add_remaining_useful_life_linear(self,df):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_number")
        max_cycle = grouped_by_unit["time"].max()
        
        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), 
                                left_on='unit_number', right_index=True)
        
        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time"]
        result_frame["RUL"] = remaining_useful_life
        
        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame

    def X_y_train_divide(self,train_df):
        # Separate X_train and y_train from train_df
        # i.e., separate RUL column from train_df
        if ('RUL' not in train_df.columns):
            train = self.add_remaining_useful_life_linear(train_df)
        else:
            train = train_df
        y_train = pd.DataFrame(train['RUL'])
        X_train = train
        X_train = X_train.drop(index_cols+settings_cols+['RUL'], axis=1)
        return X_train, y_train