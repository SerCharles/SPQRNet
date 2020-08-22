'''
绘制一个模型的训练，测试模型
'''

import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboardX import SummaryWriter

class PlotCurves:
    def GetAccuracy(self, file):
        '''
        描述：读取记录文件，获取训练测试准确率
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
        writer = SummaryWriter()
        for epoch in range(len(y_train)):
            writer.add_scalars('scalar/test', {"train" : y_train[epoch], "valid" : y_valid[epoch]}, epoch)
            #writer.add_scalar('scalar/test', y_valid, epoch)
        writer.close()
        
    def PlotContrast(self, ScalarList, length):
        writer = SummaryWriter()
        for epoch in range(length):
            TheScalar = {}
            for name in ScalarList.keys():
                TheScalar[name] = ScalarList[name][epoch]
            writer.add_scalars('scalar/test', TheScalar, epoch)
            #writer.add_scalar('scalar/test', y_valid, epoch)
        writer.close()
        
    def __init__(self, model_list):
        super(PlotCurves, self).__init__()
        ScalarList = {}
        length = -1
        if len(model_list) == 1:
            item = model_list[0]
            model_name = item["model_name"]
            model_dir = item["model_dir"]
            log_place = "../result/" + model_dir + ".txt"
            y_train, y_valid = self.GetAccuracy(log_place)
            ScalarList["train"] = y_train
            ScalarList["valid"] = y_valid
            length = len(y_train)
            self.PlotContrast(ScalarList, length)        
        else:
            for item in model_list:
                model_name = item["model_name"]
                model_dir = item["model_dir"]
                log_place = "../result/" + model_dir + ".txt"
                y_train, y_valid = self.GetAccuracy(log_place)
                if length < 0:
                    length = len(y_train)
                ScalarList[model_name] = y_valid
            #self.PlotModel(y_train, y_valid, model_name, save_place)
            self.PlotContrast(ScalarList, length)
        
if __name__ == '__main__':
    PlotModel = []
    Model0 = {"model_name":"PCN", "model_dir":"0"}
    Model10 = {"model_name":"10", "model_dir":"10"}
    Model100 = {"model_name":"100", "model_dir":"100"}
    Model1000 = {"model_name":"1000", "model_dir":"1000"}
    Model10000 = {"model_name":"10000", "model_dir":"10000"}

    PlotModel.append(Model0)
    PlotModel.append(Model10)
    PlotModel.append(Model100)
    PlotModel.append(Model1000)
    PlotModel.append(Model10000)

    PlotCurves(PlotModel)
