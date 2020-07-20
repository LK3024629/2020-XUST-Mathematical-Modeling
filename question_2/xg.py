import os
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import plot_importance

# Load data set
df=pd.read_csv('/home/lk/Documents/AI/mathmodel/Datasets/lingtong.csv')

# The data set is divided into X and y
X = df.iloc[:,1:4]
Y = df.iloc[:,4]

# Fit the model with the full data set
model = XGBClassifier()
model.fit(X, Y)

# Variable importance visualization
name_list = ['max_temp', 'min_temp', 'ave_temp']
plt.figure()
plt.bar(name_list, model.feature_importances_)
for a, b in zip(name_list, model.feature_importances_):
    plt.text(a, b+0.003, b, ha='center', va='bottom')
plt.ylabel('feature_importances')

if not os.path.exists('picture'):
    os.mkdir('picture')
plt.savefig('./picture/' +'1' + '.png', dpi=100, bbox_inches = 'tight')
plt.close('all')
# plt.show()

plot_importance(model)
if not os.path.exists('picture'):
    os.mkdir('picture')
plt.savefig('./picture/' + '2' + '.png', dpi=100, bbox_inches = 'tight')
plt.close('all')
# plt.show()