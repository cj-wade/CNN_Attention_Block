import torch
from torch import nn
from torch.autograd import Variable
import os
from utils import saveModel, loadModel, chooseData, writeHistory, writeLog, get_parameter_number
import time
from backbone.resnet_base import resnet, SE_resnet, CBMA_resnet, FA_resnet


class Net(nn.Module):
    def __init__(self, model, CLASS=102):
        super(Net, self).__init__()
        # 选择resnet 除最后一层的全连接，改为CLASS输出
        self.resnet = model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=2048, out_features=CLASS)

    def forward(self, x, train_flag='train'):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if train_flag == "train":
            x = self.drop(x)
        x = self.fc(x)
        return x


def train(modelConfig, dataConfig, logConfig):
    """
    训练
    :param modelConfig: 模型配置
    :param dataConfig: 数据配置
    :param logConfig:  日志配置
    :return:
    """
    # 模型配置
    model = modelConfig['model']
    criterion = modelConfig['criterion']
    optimzer = modelConfig['optimzer']
    epochs = modelConfig['epochs']
    device = modelConfig['device']

    # 数据加载器
    trainLoader = dataConfig['trainLoader']
    validLoader = dataConfig['validLoader']
    trainLength = dataConfig['trainLength']
    validLength = dataConfig['validLength']

    # 日志及模型保存
    modelPath = logConfig['modelPath']
    historyPath = logConfig['historyPath']
    logPath = logConfig['logPath']
    lastModelPath = logConfig['lastModelPath']

    trainLosses = []
    trainAcces = []
    validLosses = []
    validAcces = []
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('train is starting in ' + now)
    bestAcc = 0.
    best_train_Acc = 0.
    best_epoch = 0

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)

        trainLoss, trainAcc = oneEpoch_train(model, trainLoader, optimzer, criterion, device)
        validLoss, validAcc = oneEpoch_valid(model, validLoader, criterion, device)

        trainLoss = trainLoss / len(trainLoader)
        trainAcc = trainAcc / trainLength
        validLoss = validLoss / len(validLoader)
        validAcc = validAcc / validLength

        # trainLosses.append(trainLoss)
        # trainAcces.append(trainAcc)
        #
        # validLosses.append(validLoss)
        # validAcces.append(validAcc)
        # 模型验证有进步时,保存模型
        if validAcc > bestAcc:
            bestAcc = validAcc
            best_train_Acc = trainAcc
            best_epoch = epoch
            # saveModel(model,modelPath)

        # 训练日志
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        trainLog = now + " Train loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(trainLoss, 100 * trainAcc)
        validLog = now + " Valid loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(validLoss, 100 * validAcc)
        best_val_log = now + ' best val Acc is {:.4f}%\n'.format(100 * bestAcc)
        best_train_log = now + ' best train Acc is {:.4f}%\n'.format(100 * best_train_Acc)
        best_epoch_log = now + ' bestAcc is : ' + str(best_epoch)
        log = trainLog + validLog + best_train_log + best_val_log + best_epoch_log

        print(log)

        # 训练历史 每个EPOCH都覆盖一次
        # history = {
        #     'trainLosses':trainLosses,
        #     'trainAcces':trainAcces,
        #     'validLosses':validLosses,
        #     'validAcces':validAcces
        # }

        # writeLog(logPath, log)
        # writeHistory(historyPath,history)

        # 保存最新一次模型
        # saveModel(model,lastModelPath)


def oneEpoch_train(model, dataLoader, optimzer, criterion, device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param optimzer: 优化器
    :param criterion: loss计算函数
    :return: loss acc
    """
    # 模式

    model.train()
    loss = 0.
    acc = 0.
    for (inputs, labels) in dataLoader:
        # 使用某个GPU加速图像 label 计算
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度设为零，求前向传播的值
        optimzer.zero_grad()
        outputs = model(inputs, train_flag="train")
        _loss = criterion(outputs, labels)

        # 反向传播
        _loss.backward()
        # 更新网络参数
        optimzer.step()

        _, preds = torch.max(outputs.data, 1)
        loss += _loss.item()
        acc += torch.sum(preds == labels).item()

    return loss, acc


def oneEpoch_valid(model, dataLoader, criterion, device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param criterion: loss计算函数
    :return: loss acc
    """
    with torch.no_grad():
        model.eval()
        loss = 0.
        acc = 0.
        for (inputs, labels) in dataLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs, train_flag="val")
            _loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            loss += _loss.item()
            acc += torch.sum(preds == labels).item()

    return loss, acc


def _stanfordDogs():
    """
     StanfordDogs数据集
     :return:
     """

    # 定义模型 定义评价 优化器等
    lr = 1e-4
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(SE_resnet.resnet50(pretrained=True), 120)

    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        # if any([pname.endswith(k) for k in ['seblock']]):
        #     attention_params += [p]
        if ('seblock' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'Cifar_resnet_base' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        else:
            p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    # print("attention:")
    # for i in attention_params:
    #     print(i.size())
    # print("classfier")
    # for i in classifier_params:
    #     print(i.size())
    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    # backbone_params = model.children()[:-3].parameters()
    # attention_classfication_params = model.children()[-3:].parameters()
    # backbone_params = list(map(id, model.Cifar_resnet_base.parameters()))
    # attention_classfication_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 10},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=0.0001
    )

    # torch.optim.lr_scheduler.StepLR(optimzer, 10, gamma=0.94, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=10)
    epochs = 150
    batchSize = 16
    worker = 2
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        # T.RandomCrop(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDDOGS', batchSize,worker, trainTransforms,testTransforms)

    # trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDDOGS', batchSize,
    #                                                                                         worker)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanforddogs.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanforddogs_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_stanforddogs.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_stanforddogs.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)

def _stanfordCars():
    """
     StanfordCars数据集
     :return:
     """

    # 定义模型 定义评价 优化器等
    lr = 1e-4
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(resnet.resnet50(pretrained=True), 196)
    # model = Net(SE_resnet.resnet50(pretrained=True), 196)
    # model = Net(CBMA_resnet.resnet50(pretrained=True), 196)
    # model = Net(FA_resnet.resnet50(pretrained=True), 196)




    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        # if any([pname.endswith(k) for k in ['seblock']]):
        #     attention_params += [p]
        if ('seblock' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'Cifar_resnet_base' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        else:
            p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    # print("attention:")
    # for i in attention_params:
    #     print(i.size())
    # print("classfier")
    # for i in classifier_params:
    #     print(i.size())
    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    # backbone_params = model.children()[:-3].parameters()
    # attention_classfication_params = model.children()[-3:].parameters()
    # backbone_params = list(map(id, model.Cifar_resnet_base.parameters()))
    # attention_classfication_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 10},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=0.0001
    )

    # torch.optim.lr_scheduler.StepLR(optimzer, 10, gamma=0.94, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=10)
    epochs = 150
    batchSize = 64
    worker = 2
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        # T.RandomCrop(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDCARS', batchSize,worker, trainTransforms,testTransforms)

    # trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDCARS', batchSize,
    #                                                                                         worker)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanfordcars.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanfordcars_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_stanfordcars.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_stanfordcars.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)

def _CUB200():
    """
    CUB200数据集
    :return:
    """

    # 定义模型 定义评价 优化器等
    lr = 1e-4
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Net(SE_resnet.resnet50(pretrained=True), 200)
    # model = Net(CBMA_resnet.resnet50(pretrained=True), 200)
    model = Net(FA_resnet.resnet50(pretrained=True), 200)

    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        # if any([pname.endswith(k) for k in ['seblock']]):
        #     attention_params += [p]
        if ('seblock' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'Cifar_resnet_base' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        else:
            p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    # print("attention:")
    # for i in attention_params:
    #     print(i.size())
    # print("classfier")
    # for i in classifier_params:
    #     print(i.size())
    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    # backbone_params = model.children()[:-3].parameters()
    # attention_classfication_params = model.children()[-3:].parameters()
    # backbone_params = list(map(id, model.Cifar_resnet_base.parameters()))
    # attention_classfication_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 10},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=0.0001
    )

    # torch.optim.lr_scheduler.StepLR(optimzer, 10, gamma=0.94, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=10)
    epochs = 150
    batchSize = 16
    worker = 2
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testTransforms = T.Compose([
        # T.Scale((550, 550)),
        T.Resize(256),
        # T.RandomCrop(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CUB200', batchSize,worker, trainTransforms,testTransforms)

    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_CUB200.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_CUB200.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)

def _Cifar_10():
    """
     Cifar-10 数据集
     :return:
     """
    seed = 0
    # 定义模型 定义评价 优化器等
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    lr = 1e-2
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Net(resnet.resnet18(pretrained=False), 10)
    # model = Net(SE_resnet.resnet18(pretrained=False), 10)
    #model = Net(CBMA_resnet.resnet18(pretrained=False), 10)
    model = Net(FA_resnet.resnet18(pretrained=False), 10)

    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        if ('cbam' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'resnet' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        # else:
        #     p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.90, weight_decay=1e-4)
    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 1},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=1e-4
    )

    torch.optim.lr_scheduler.StepLR(optimzer, 5, gamma=0.94, last_epoch=-1)
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=20)
    epochs = 250
    batchSize = 256
    worker = 4
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        T.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        T.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    testTransforms = T.Compose([
        # T.Resize(550),
        # T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    # 加载数据集
    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CIFAR_10', batchSize,
                                                                                            worker, trainTransforms,
                                                                                            testTransforms)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_cifar_10.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_cifar_10_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_cifar_10.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_cifar_10.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    # 训练（包含一次训练和验证）
    train(modelConfig, dataConfig, logConfig)


def _Cifar_100():
    """
     Cifar-100 数据集
     :return:
     """
    seed = 0
    # 定义模型 定义评价 优化器等
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    lr = 1e-2
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Net(resnet.resnet18(pretrained=False), 100)
    # model = Net(SE_resnet.resnet18(pretrained=False), 100)
    model = Net(CBMA_resnet.resnet18(pretrained=False), 100)
    # model = Net(FA_resnet.resnet18(pretrained=False), 100)

    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        if ('cbam' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'resnet' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        # else:
        #     p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.90, weight_decay=1e-4)
    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 1},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=1e-4
    )

    # torch.optim.lr_scheduler.StepLR(optimzer, 50, gamma=0.1, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=20)
    epochs = 250
    batchSize = 256
    worker = 4
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        T.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        T.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    testTransforms = T.Compose([
        # T.Resize(550),
        # T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CIFAR_100', batchSize,
                                                                                            worker, trainTransforms,
                                                                                            testTransforms)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_cifar_100.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_cifar_100_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_cifar_100.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_cifar_100.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)


def _Imagenet_1K():
    """
     Cifar-10 数据集
     :return:
     """
    seed = 0
    # 定义模型 定义评价 优化器等
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    lr = 1e-2
    print("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Net(resnet.resnet18(pretrained=False), 10)
    # model = Net(SE_resnet.resnet18(pretrained=False), 10)
    model = Net(CBMA_resnet.resnet18(pretrained=False), 1000)
    # model = Net(FA_resnet.resnet18(pretrained=False), 10)

    all_params = model.parameters()
    attention_params = []
    classifier_params = []
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
        print(pname)
        if ('cbam' in pname):
            attention_params += [p]
            # p.requires_grad = False
        elif ('fc' in pname and 'resnet' not in pname):
            classifier_params += [p]
            # p.requires_grad = False
        # else:
        #     p.requires_grad = False
        # 取回分组参数的id

    print(get_parameter_number(model))

    params_id = list(map(id, attention_params)) + list(map(id, classifier_params))
    # 取回剩余分特殊处置参数的id
    backbone_params = list(filter(lambda p: id(p) not in params_id, all_params))

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimzer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.90, weight_decay=1e-4)
    optimzer = torch.optim.SGD(
        [
            {'params': backbone_params, 'lr': lr * 1},
            {'params': attention_params, 'lr': lr * 1},
            {'params': classifier_params, 'lr': lr * 1},
        ],
        lr=lr, momentum=0.9, weight_decay=1e-4
    )

    # torch.optim.lr_scheduler.StepLR(optimzer, 50, gamma=0.1, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=20)
    epochs = 250
    batchSize = 256
    worker = 4
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        T.Resize(256),
        T.RandomRotation(15),
        # T.RandomResizedCrop(224,scale=(0.85,1.15)),
        T.RandomCrop(224),
        T.ToTensor(),
        normalize
    ])

    testTransforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('ImageNet-1k', batchSize,
                                                                                            worker, trainTransforms,
                                                                                            testTransforms)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(),  'checkpoints', '_ImageNet-1k.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_ImageNet-1k_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_ImageNet-1k.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_ImageNet-1k.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)


if __name__ == '__main__':
    print(torch.__version__)
    # _stanfordDogs()
    _stanfordCars()
    # _CUB200()
    # _Cifar_10()
    # _Cifar_100()
    # _Imagenet_1K()

# 已执行命令


# 待执行命令
