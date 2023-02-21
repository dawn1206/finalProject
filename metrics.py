import torch
import sklearn
import torchmetrics
import pandas as pd
device = torch.device("cpu")
def test_loop(dataloader, model, loss_fn,epoch,df,modelName):
    model.eval()
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    test_recall = torchmetrics.Recall(task="multiclass", average='none', num_classes=4)
    test_precision = torchmetrics.Precision(task="multiclass", average='none', num_classes=4)
    test_auc = torchmetrics.AUROC(task="multiclass", average="macro", num_classes=4)
    test_f1 = torchmetrics.classification.MultilabelF1Score(num_labels=4,)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).to(device)
            y =y.long().to(device)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 一个batch进行计算迭代
            test_acc(pred.argmax(1), y)
            test_auc.update(pred, y)
            test_recall(pred.argmax(1), y)
            test_precision(pred.argmax(1), y)
            zero_arr = torch.zeros(y.shape[0], 4)
            zero_arr[torch.arange(y.shape[0]), y] = 1
            test_f1(pred, zero_arr)

    test_loss /= num_batches
    # correct /= size

    # 计算一个epoch的accuray、recall、precision、AUC
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    f1Score = test_f1.compute()
    df.loc[epoch]["loss"] = test_loss
    df.loc[epoch]["acc"] = total_acc.item()
    df.loc[epoch]["recall"] = total_recall.mean().item()
    df.loc[epoch]["precision"] = total_precision.mean().item()
    df.loc[epoch]["auc"] = total_auc.item()
    df.loc[epoch]["f1"] = f1Score.item()

    with open(modelName+".log","a") as f:
        # f.writelines(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, \n"
        #   f"Avg loss: {test_loss:>8f},\n "
        #   f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
        f.writelines("Test epoch: " + str(epoch) + "\n")
        # f.write("Accuracy: " + str(100 * correct)+"% \n")
        f.write("Avg loss: "+str(test_loss)+"\n ")
        f.write("torch metrics acc:"+str(100 * total_acc)+"%\n")
        f.write("recall of every test dataset class: " + str(total_recall)+"\n")
        f.write("precision of every test dataset class: "+ str(total_precision)+"\n")
        f.write("auc:"+ str(total_auc.item())+"\n")
        f.write("f1: "+str(f1Score.data)+"\n")
        f.write('-' * 89)

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()
    return test_loss,test_acc
