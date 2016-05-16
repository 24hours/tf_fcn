import numpy as np
import scipy

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(prediction, label, class_num=21):
    hist = np.zeros((class_num, class_num))
    loss = 0
    for idx in range(prediction.shape[0]):
        hist += fast_hist(  np.reshape(label[idx,:,:], [-1]),
                            np.reshape(prediction[idx,:,:], [-1]),
                            class_num)

    return hist

def pixel_accuracy(prediction, label):
    p = np.reshape(prediction, [-1])
    l = np.reshape(label, [-1])
    acc = p == l
    acc_not_background = acc[l!=0]

    return np.sum(acc_not_background)/len(p)

def summary(prediction):
    return np.mean(prediction)

def test(prediction, label):
    hist = compute_hist(prediction, label)
    accuracy = pixel_accuracy(prediction, label)
    summa = summary(prediction)
    # overall accuracy
    print '>>> Summary', summa
    print '>>> Pixel accuracy', accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>> overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>> mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>> mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>> fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
