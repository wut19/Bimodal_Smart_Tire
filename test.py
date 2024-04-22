import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

y_true = [1, 1, 1, 0, 0, 0, 2, 2, 2, 2]
y_pred = [1, 0, 0, 0, 2, 1, 0, 0, 2, 2]

print(classification_report(y_true, y_pred))
print(recall_score(y_true, y_pred, average='macro'))

# classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
 
# gt_labels = np.zeros(1000).reshape(10, -1)
# for i in range(10):
#     gt_labels[i] = i
# gt_labels = gt_labels.reshape(1, -1).squeeze()
# print("gt_labels.shape : {}".format(gt_labels.shape))
# print("gt_labels : {}".format(gt_labels[::5]))
 
# pred_labels = np.zeros(1000).reshape(10, -1)
# for i in range(10):
#     pred_labels[i] = np.random.randint(0, i + 1, 100)
# pred_labels = pred_labels.reshape(1, -1).squeeze()
# print("pred_labels.shape : {}".format(pred_labels.shape))
# print("pred_labels : {}".format(pred_labels[::5]))
 
# confusion_mat = confusion_matrix(gt_labels, pred_labels)
# print("confusion_mat.shape : {}".format(confusion_mat.shape))
# print("confusion_mat : {}".format(confusion_mat))
 
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat/100., display_labels=classes)
# disp.plot(
#     include_values=True,            
#     cmap=plt.cm.Blues,                 
#     ax=None,                        
#     xticks_rotation="vertical",   
#     values_format=".0%",
# )
# label_font = {'size':'18'}  # Adjust to fit
# disp.ax_.set_xlabel('Predicted labels', fontdict=label_font)
# disp.ax_.set_ylabel('Observed labels', fontdict=label_font)
# disp.ax_.set_title('Confusion Matrix (%)')
# plt.show()