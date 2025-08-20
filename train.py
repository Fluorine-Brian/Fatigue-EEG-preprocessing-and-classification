import torch
import scipy.io as sio
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
from model import SECNN
import os
import logging
from args import build_args

torch.cuda.empty_cache()
torch.manual_seed(0)


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, filename='checkpoint'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.filename = filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.filename + '.pt')
        logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')


def run(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # 设置日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = os.path.join(args.log_dir, f't{timestamp}_e{args.n_epoch}_p{args.patience}.txt')
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 加载数据
    tmp = sio.loadmat(args.data_dir)
    xdata = np.array(tmp['samples'])
    ydata = np.array(tmp['labels']).flatten()

    # 数据集划分
    x_train, x_temp, y_train, y_temp = train_test_split(xdata, ydata, test_size=0.4, stratify=ydata, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    def log_data_distribution(name, labels):
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"{name} data distribution: {distribution}")
        logging.info(f"{name} data distribution: %s", distribution)

    log_data_distribution("Training", y_train)
    log_data_distribution("Validation", y_val)
    log_data_distribution("Test", y_test)

    # 调整数据维度
    x_train = x_train.reshape(x_train.shape[0], 1, 3, 250)
    x_val = x_val.reshape(x_val.shape[0], 1, 3, 250)
    x_test = x_test.reshape(x_test.shape[0], 1, 3, 250)

    my_net = SECNN(
        classes = args.classes,
        sampleChannel = args.sampleChannel,
        sampleLength = args.sampleLength,
        N1 = args.N1,
        d = args.d,
        kernelLength = args.kernelLength,
        reduction = args.reduction,
        dropout_rate = args.dropout_rate
        ).double().to(device)
    optimizer = optim.Adam(my_net.parameters(), lr=args.lr)
    loss_class = torch.nn.CrossEntropyLoss().to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, filename='checkpoint')

    # 训练过程
    for epoch in range(args.n_epoch):
        my_net.train()
        running_loss = 0.0
        for i in range(0, len(x_train), args.batch_size):
            inputs = torch.DoubleTensor(x_train[i:i + args.batch_size]).to(device)
            labels = torch.LongTensor(y_train[i:i + args.batch_size]).to(device)

            optimizer.zero_grad()
            class_output = my_net(inputs)
            loss = loss_class(class_output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证
        my_net.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            val_inputs = torch.DoubleTensor(x_val).to(device)
            val_labels = torch.LongTensor(y_val).to(device)
            val_output = my_net(val_inputs)
            val_loss = loss_class(val_output, val_labels)
            val_running_loss = val_loss.item()

        early_stopping(val_running_loss, my_net)

        print(f"Epoch [{epoch + 1}/{args.n_epoch}], Loss: {running_loss / (len(x_train) / args.batch_size):.6f}, Val Loss: {val_running_loss:.6f}")
        logging.info(f"Epoch [{epoch + 1}/{args.n_epoch}], Loss: {running_loss / (len(x_train) / args.batch_size):.6f}, Val Loss: {val_running_loss:.6f}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 测试
    # my_net.load_state_dict(torch.load(args.best_checkpoint))
    my_net.eval()
    with torch.no_grad():
        test_inputs = torch.DoubleTensor(x_test).to(device)
        test_labels = torch.LongTensor(y_test).to(device)
        test_output = my_net(test_inputs)
        preds = test_output.argmax(axis=1).cpu().numpy()

        # 计算评估指标
        acc = accuracy_score(test_labels.cpu().numpy(), preds)
        pre = precision_score(test_labels.cpu().numpy(), preds, average='macro')
        rec = recall_score(test_labels.cpu().numpy(), preds, average='macro')
        f1 = f1_score(test_labels.cpu().numpy(), preds, average='macro')  # 计算 F1 分数

        # 输出在同一行
        log_msg = (f'Epoch [{args.n_epoch}], accuracy: {acc:.4f}, precision: {pre:.4f}, '
                   f'recall: {rec:.4f}, F1 score: {f1:.4f}')
        print(log_msg)  # 控制台输出
        logging.info(log_msg)  # 日志输出

    log_filename_with_acc = os.path.join(args.log_dir, f't{timestamp}_e{args.n_epoch}_p{args.patience}_a{acc:.4f}_f1{f1:.4f}.txt')
    os.rename(log_filename, log_filename_with_acc)


if __name__ == '__main__':
    args = build_args()
    run(args)
