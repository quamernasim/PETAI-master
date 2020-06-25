import numpy as np
import pandas as pd
import os


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Acc: ': acc,
                'Class Accuracy: ': acc_cls,
                'Mean Class Acc: ': mean_acc_cls,
                'Freq Weighted IoU: ': fwavacc,
                'Mean IoU: ': mean_iu,
                'confusion_matrix': self.confusion_matrix}, cls_iu
    def get_classification_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc_cls = np.nanmean(acc_cls)
        prec = np.diag(hist).sum()/(np.diag(hist).sum()+ hist.sum(axis=0)-np.diag(hist))
        recall = np.diag(hist).sum()/(np.diag(hist).sum()+ hist.sum(axis=1)-np.diag(hist))
        f1 = 2*(prec*recall)/(prec+recall)

        #True positive: diagonal position, cm(x, x).
        #False positive: sum of column x (without main diagonal), sum(cm(:, x))-cm(x, x).
        #False negative: sum of row x (without main diagonal), sum(cm(x, :), 2)-cm(x, x).


        return {'Class Accuracy: ': acc_cls,
                'Mean Class Acc: ': mean_acc_cls,
                'Prec: ': prec,
                'Recall: ': recall,
                'F1 Score: ':f1,
                'confusion_matrix': self.confusion_matrix}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def save_classification_csv(score, val_fname, epoch, domain, n_classes):

    pd.DataFrame([score['Mean Class Acc: ']], columns = ["MCA"]).to_csv(os.path.join(val_fname, "metrics", domain + "_mca.csv"), index=False, mode='a', header=(epoch==0))

    cname = os.path.join(val_fname, "metrics", "confusion_matrix", domain + "_confusion_matrix_" + str(epoch + 1) + ".csv")
    pd.DataFrame(score["confusion_matrix"]).to_csv(cname, index=False)

    pd.DataFrame(score["Class Accuracy: "].reshape((1, n_classes)), columns=list(range(n_classes))).to_csv(os.path.join(val_fname,"metrics", domain + "_class_acc.csv"), index=False, mode = "a", header = (epoch == 0))
    pd.DataFrame(score["Prec: "].reshape((1, n_classes)), columns=list(range(n_classes))).to_csv(os.path.join(val_fname,"metrics", domain + "_prec.csv"), index=False, mode = "a", header = (epoch == 0))
    pd.DataFrame(score["Recall: "].reshape((1, n_classes)), columns=list(range(n_classes))).to_csv(os.path.join(val_fname,"metrics", domain + "_recall.csv"), index=False, mode = "a", header = (epoch == 0))
    pd.DataFrame(score["F1 Score: "].reshape((1, n_classes)), columns=list(range(n_classes))).to_csv(os.path.join(val_fname,"metrics", domain + "_f1.csv"), index=False, mode = "a", header = (epoch == 0))