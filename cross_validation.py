def acc(sum_acc, this_fold_accuracy, fold):
    print('---------------------------------------------')
    sum_acc += this_fold_accuracy
    print('5-CV Average Validation Set Accuracy %g' % (sum_acc/(fold+1)))
    print('---------------------------------------------')
    return sum_acc
