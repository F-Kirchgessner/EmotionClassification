import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from src.data_utils import get_Dataset, OverfitSampler, get_pics
from src.classifiers.simple_emo_classifier import SimpleEmoClassifier
#from src.classifiers.landmark_emo_classifier import LandmarkEmoClassifier
from src.solver import Solver

import time
import datetime
import sys
import os

def train():
    # path of this file
    ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Year-month-day_Hour-Minute-Second
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Loading data...")

    train_data, val_data = get_Dataset()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=25, shuffle=False, num_workers=10)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size = 5, shuffle = False, num_workers = 2, sampler = OverfitSampler(50))
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=False,num_workers=2, sampler=OverfitSampler(20))

    log_n = 10
    epochs = 10

    model = SimpleEmoClassifier(weight_scale=0.0005)
    solver = Solver(optim_args={'lr': 5e-5})
    tic = time.time()
    solver.train(model, train_loader, val_loader, num_epochs=epochs, log_nth=log_n)
    temp_time = time.time() - tic
    m, s = divmod(temp_time, 60)
    h, m = divmod(m, 60)
    print('Done after %dh%02dmin%02ds' % (h, m, s))

    # Save model
    model.save("models/model_{}.model".format(timestamp))

    plt.subplot(2, 1, 1)
    plt.plot(solver.train_loss_history, '-', label='train_loss')
    x = np.linspace(0, len(solver.train_loss_history), len(solver.val_loss_history))
    plt.plot(x, solver.val_loss_history, '-o', label='val_loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation loss | %d Epochs' % epochs)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o', label='train_acc=%.4f' % (solver.train_acc_history[-1]))
    plt.plot(solver.val_acc_history, '-o', label='val_acc=%.4f' % (solver.val_acc_history[-1]))
    plt.legend(loc='upper left')
    plt.title('Training vs Validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.gca().yaxis.grid(True)

    plt.gcf().set_size_inches(15, 15)
    plt.tight_layout()
    plt.savefig(ABS_PATH + 'output/performance_{}.png'.format(timestamp))
    plt.gcf().clear()

    # plot examples:
    model.eval()

    # get_pics might not work! If it doesn't, uncomment the old code.
    test_pics, example_labels, filenames, amount_example_pics = get_pics(train_data, val_data)
    output = model.forward(Variable(torch.Tensor(test_pics).float()).cuda())
    emotions = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}
    print('0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise')
    print(np.argmax(output.data.cpu().numpy(), axis=1, out = np.empty(amount_example_pics, dtype='int64')))
    print(example_labels)
    output = torch.nn.functional.softmax(output).cpu().data.numpy()

    # plot images and write output under them, very unsure!! Better check on this one!
    for i in range(amount_example_pics):
        plt.subplot(amount_example_pics, 1, i + 1)
        #plt.legend(loc='upper left')
        plt.title('%s: Truth=%s, N=%.2e, A=%.2e, C=%.2e, D=%.2e, F=%.2e, H=%.2e, Sad=%.2e, Sur=%.2e' % (filenames[i], emotions[example_labels[i]], list(output[i])[0], list(output[i])[1], list(
            output[i])[2], list(output[i])[3], list(output[i])[4], list(output[i])[5], list(output[i])[6], list(output[i])[7]))
        plt.imshow(test_pics[i][0])

    plt.tight_layout()
    plt.savefig(ABS_PATH + 'output/examples_{}.png'.format(timestamp))
    plt.gcf().clear()


if __name__ == "__main__":
    train()
