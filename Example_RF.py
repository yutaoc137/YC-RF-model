import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# read the data and print the first 5 rows
df = pd.read_csv("~/Desktop/data.csv")
# print(df.head(5))

# trim the data
df_trimmed = df.apply(lambda x: x.str.strip('"') if x.dtype == 'object' else x)

# print(df_trimmed.dtypes)
# exit(0)


# Filter rows with missing values
survey_data = df_trimmed.dropna(axis=0)

# Choose target and features
y = survey_data.Mode
selected_features = ['Sect_0_time', 'Sect_1_time', 'Sect_2_time', 'Sect_3_time',
                     'Sect_4_time', 'Sect_5_time', 'Sect_6_time', 'Sect_7_time']
X = survey_data[selected_features]

# split data into training and testing data.
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# train the model
forest_model = RandomForestClassifier(n_estimators=300, oob_score=True, max_features='auto')
forest_model.fit(train_X, train_y)
mode_preds = forest_model.predict(test_X)

# test results
print("Accuracy:", metrics.accuracy_score(test_y, mode_preds))
print(f'Out-of-bag score estimate: {forest_model.oob_score_:.3}')
print(confusion_matrix(test_y, mode_preds))
print(classification_report(test_y, mode_preds))
print(accuracy_score(test_y, mode_preds))

# ROC-AUC curve
mode_preds = forest_model.predict_proba(test_X)
preds = mode_preds[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_y, preds, pos_label='2')
roc_auc = metrics.auc(fpr, tpr)

# plot roc_auc curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
