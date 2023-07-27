from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize


fd_path1 = r'C:\Users\g\Desktop\QC\QC\1_val.csv'
fd_path2 = r'C:\Users\g\Desktop\QC\QC\2_val.csv'
fd_path3 = r'C:\Users\g\Desktop\QC\QC\3_val.csv'
a1 = pd.read_csv(fd_path1, encoding='utf-8')
a2 = pd.read_csv(fd_path2, encoding='utf-8')
a3 = pd.read_csv(fd_path3, encoding='utf-8')
fd = pd.concat([a1, a2, a3], axis=0)
y_true = fd['label']-1
y_pred = fd['predict_result']-1



fd_path11 = r'C:\Users\g\Desktop\QC\QC\roc\1_val.csv'
fd_path21 = r'C:\Users\g\Desktop\QC\QC\roc\2_val.csv'
fd_path31 = r'C:\Users\g\Desktop\QC\QC\roc\3_val.csv'
a11 = pd.read_csv(fd_path11, encoding='utf-8')
a21 = pd.read_csv(fd_path21, encoding='utf-8')
a31 = pd.read_csv(fd_path31, encoding='utf-8')
fd1 = pd.concat([a11, a21, a31], axis=0)

y_true1 = y_true.values



y_pred1 = fd1.iloc[:,2:5].values
# y_true1 = label_binarize(y_true1, classes=list(range(len(y_pred1[0]))))



cm = confusion_matrix(y_true=y_true, y_pred=y_pred)


# 提取真正例（True Positives，TP）
tp = np.diag(cm)
print("True Positives (TP):", tp)

# 提取真反例（True Negatives，TN）
tn = np.sum(cm) - np.sum(cm, axis=0) - np.sum(cm, axis=1) + tp
print("True Negatives (TN):", tn)

# 提取假正例（False Positives，FP）
fp = np.sum(cm, axis=0) - tp
print("False Positives (FP):", fp)

# 提取假反例（False Negatives，FN）
fn = np.sum(cm, axis=1) - tp
print("False Negatives (FN):", fn)

actual_positives = np.sum(cm, axis=1)


# 计算特异度（Specificity/True Negative Rate，TNR）
specificity = tn / (tn + fp)

# 计算准确率（Accuracy）
accuracy = accuracy_score(y_true, y_pred)

# 计算AUC
# auc = roc_auc_score(y_true, y_pred,multi_class='ovo')
# auc = roc_auc_score(y_true1, y_pred1, average='macro')
# 计算F1值
f1 = f1_score(y_true, y_pred, average='weighted')

# 计算精确率（Precision）
precision = precision_score(y_true, y_pred, average='weighted')

# 计算召回率（Recall）
recall = recall_score(y_true, y_pred, average='weighted')

# 打印计算结果
print("Specificity (TNR):", specificity)
print("Accuracy:", accuracy)
# print("AUC:", auc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)





# 计算每个类别的TPR和FPR
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_classes = len(np.unique(y_true))
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred1[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # 绘制ROC曲线
# plt.figure()
# colors = ['red', 'blue', 'green']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2, label=str(i+1) + '(area = %0.4f)' % roc_auc[i])
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc='lower right')
# plt.show()



# target_names = ['grade0', 'grade1','grade2','grade3','grade4']
# # 绘图格式
# plot_confusion_matrix(cm, target_names)  # 调用函数绘制混淆矩阵
# plt.show()




# 获取类别数量
num_classes = cm.shape[0]

# 绘制混淆矩阵图像
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# 设置轴标签
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [1,2,3])
plt.yticks(tick_marks, [1,2,3])

# 添加数值标签
thresh = cm.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# 设置其他绘图属性
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# 显示混淆矩阵图像
plt.show()