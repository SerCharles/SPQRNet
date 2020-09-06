'''
Description: Visualizing the results. After running the program, input
                tensorboard --logdir runs
            to see the result
Author:Charles Shen
Date:8/23/2020
'''

import os
import argparse
import numpy as np
from tensorboardX import SummaryWriter

class PlotCurves:
    def GetAccuracy(self, file):
        '''
            Description: Input the result file
            input: file dir
            output: train and test results, in list
        '''
        y_train = []
        y_valid = []
        with open(file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                item = lines[i]
                item_split = item.split()
                if(len(item_split) <= 0):
                    continue
                number = float(item_split[0])
                if i % 2 == 0:
                    y_train.append(number)
                else:
                    y_valid.append(number)
        return y_train, y_valid            


    def PlotTensorboard(self, y_train, y_valid):
        '''
            Description: print the train and valid curve of a model
            input: train results, valid results
            output: no
        '''
        writer = SummaryWriter()
        for epoch in range(len(y_train)):
            writer.add_scalars('scalar/test', {"train" : y_train[epoch], "valid" : y_valid[epoch]}, epoch)
            #writer.add_scalar('scalar/test', y_valid, epoch)
        writer.close()
        
    def PlotContrast(self, ScalarList, length):
        '''
            Description: print the valid curves of a model
            input: the Scalar list(ScalarList[name][epoch] means the result of the name of the model at the epoch), epoch nums
            output: no
        '''
        writer = SummaryWriter()
        for epoch in range(length):
            TheScalar = {}
            for name in ScalarList.keys():
                TheScalar[name] = ScalarList[name][epoch]
            writer.add_scalars('scalar/test', TheScalar, epoch)
            #writer.add_scalar('scalar/test', y_valid, epoch)
        writer.close()
        
    def __init__(self, result_dir):
        '''
            Description: main function
            input: the result dir, which contains many txts indicating the result
            output: no
        '''
        super(PlotCurves, self).__init__()
        length = -1
        ScalarList = {}
        for file_name in os.listdir(result_dir): 
            split_name = file_name.split('.')
            if len(split_name) == 2 and split_name[-1] == 'txt':
                file_path = os.path.join(result_dir, file_name)
                model_name = split_name[0]
                y_train, y_valid = self.GetAccuracy(file_path)
                min_valid = round(min(y_valid), 2)
                print(model_name, min_valid)
                if length < 0:
                    length = len(y_train)
                ScalarList[model_name] = y_valid
        self.PlotContrast(ScalarList, length)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Plot')
    parser.add_argument('--base', type = str, default = '../result', help='result base dir')
    parser.add_argument('--dir', type = str, default = 'test', help='result specific dir')
    args = parser.parse_args()
    result_place = os.path.join(args.base, args.dir)
    PlotCurves(result_place)