import numpy as np
def result_Summary(guess_i, label, isPrint=False):
    guess_i = np.array(guess_i)
    label = (label.data).cpu().numpy()


    TP = 0
    FP = 0
    FN = 0
    TN = 0

    correct = np.sum((guess_i) == (label))
    for idx in range(len(guess_i)):
        if str(label[idx]) == '1':
            if str(guess_i[idx]) == '1':
                TP += 1
            else:
                FN += 1
        else:
            if str(guess_i[idx]) == '1':
                FP += 1
            else:
                TN += 1
                
    if isPrint:
        print '                   TP : ', TP, ' FP : ', FP, ' FN : ', FN, ' TN : ', TN
    return TP, FP, FN, TN


def result_correct(guess_i, label, isPrint=False):
    guess_i = np.array(guess_i)
    label = (label.data).cpu().numpy()
    correct = np.sum((guess_i) == (label))
    if isPrint:
        print '                   Accuracy : ', correct ,'/', label.shape[0], '----->', (correct * 100 / label.shape[0]) , '%'
    return correct
