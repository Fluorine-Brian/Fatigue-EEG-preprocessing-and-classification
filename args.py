import argparse


def build_args():
    parser = argparse.ArgumentParser(description="Fatigue Detection")

    # training arguments
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--data_dir', type=str, default='C:/算法源代码及结果\清洗后数据/qualified_dataset.mat',
                        help='Path to the dataset (.mat file)')
    # parser.add_argument('--best_checkpoint', type=str, default='./bestcheckpoint0.938613.pt',
    #                     help='File name for the best checkpoint')

    # model parameters
    parser.add_argument('--classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--sampleChannel', type=int, default=3, help='Number of EEG channels')
    parser.add_argument('--sampleLength', type=int, default=250, help='Length of EEG samples')
    parser.add_argument('--N1', type=int, default=16, help='Number of filters in the first layer')
    parser.add_argument('--d', type=int, default=8, help='Depth of the network')
    parser.add_argument('--kernelLength', type=int, default=16, help='Length of the convolutional kernel')
    parser.add_argument('--reduction', type=int, default=4, help='Reduction ratio for SE block')
    parser.add_argument('--dropout_rate', type=float, default=0.7, help='Dropout rate')
    args = parser.parse_args()
    return args
