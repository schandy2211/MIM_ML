import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,  mean_absolute_error
from util import extract_feature_vector
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance

# Use the extract_feature_vector() function to generate the feature vector
csv_dir = 'xtb'
csv_dir1 = 'xtb'
csv_dir_test = 'xtb_test'
csv_dir_test1 = 'xtb_test'
pdb_dir = 'pdb'
pdb_dir_test = 'pdb_test'
chem_dir_test = 'pdb_test'
chem_dir = 'pdb'
# Create an empty list to store the dataframes for each PDB file
data = []
# Loop through each PDB file in the directory

for pdb_dir_current in [pdb_dir, pdb_dir_test]:
    for pdb_file in os.listdir(pdb_dir_current):
        if pdb_file.endswith('.pdb'):
            # Define the path to the corresponding chem_shift file
            chem_shift_file = os.path.join(pdb_dir_current, pdb_file[:-4] + '.txt')

            # Find the corresponding gbsa.out file in csv_dir
            gbsa_file = None
            for csv_current in [csv_dir, csv_dir_test]:
                for file in os.listdir(csv_current):
                    if file.startswith(pdb_file[:-4]) and file.endswith('gbsa.csv'):
                        gbsa_file = os.path.join(csv_current, file)
                        break

            # Find the corresponding xtb.out file in csv_dir1
            xtb_file = None
            for csv_current1 in [csv_dir1, csv_dir_test1]:
                for file in os.listdir(csv_current1):
                    if file.startswith(pdb_file[:-4]) and file.endswith('xtb.csv'):
                        xtb_file = os.path.join(csv_current1, file)
                        break

            feature_vector = extract_feature_vector(os.path.join(pdb_dir_current, pdb_file), chem_shift_file, gbsa_file, xtb_file)

            # Convert the feature vector to a list of dictionaries, where each dictionary corresponds to one row of the dataframe
            for k, v in feature_vector.items():
                row = {'name': v['name'], 'pdb_id':v['pdb_id'], 'mw': v['mw'], 'an': v['an'],'en': v['en'], 'resname': v['resname'], 'chem_shift': v['chem_shift'], 'rad': v['rad'], 'hbond': v['hbond'], 'SASA': v['SASA'], 'q': v['q'], 'AA': v['AA'], 'Alpha': v['Alpha'], 'CN': v['CN'], 'SASA_squared': v['SASA']**2, 'rad_squared': v['rad']**2, 'SASA_rad_product': v['SASA'] * v['rad'], 'CN_squared': v['CN']**2}
                for i in range(len(v['neighbor_names'])):
                     row[f'neighbor_names_{i}'] = v['neighbor_names'][i]
                     row[f'neighbor_resnames_{i}'] = v['neighbor_resnames'][i]
                data.append(row)
# Create a dataframe from the list of dictionaries
df = pd.DataFrame(data)

# add function-generated features
df1 =df[(df.iloc[:, 0].astype(str).str.startswith('H'))] # 
df1 = pd.get_dummies(df1, columns=['name', 'resname'] + [f'neighbor_names_{i}' for i in range(len(v['neighbor_names']))] + [f'neighbor_resnames_{i}' for i in range(len(v['neighbor_resnames']))])
df1['name_7all'] = df1[['name_H71', 'name_H72', 'name_H73']].apply(lambda x: x.sum(), axis=1)
df1 = df1.drop(columns=['name_H71', 'name_H72', 'name_H73'])
df1['name_68'] = df1[['name_H6','name_H8', 'name_H61', 'name_H62']].apply(lambda x: x.sum(), axis=1)
df1 = df1.drop(columns=['name_H61', 'name_H62'])

structures_to_exclude = ['5IZP_min', '5ZLD_min', '2M3P_min', '1JS7_min', '1K8L_min', '5UZF_min', '6DM7_min', '7BFX_min']
df2 = df1[(df1.iloc[:, 0].astype(str).str.startswith('5IZP_min'))]
#df3 = df1[(~df1.iloc[:, 0].astype(str).str.startswith(''))]
df3 = df1[~df1.iloc[:, 0].astype(str).isin(structures_to_exclude)]
# Save the DataFrame to a CSV file
#df.to_csv('feature_vector_all.csv', index=False)
df3.to_csv('feature_vector_H.csv', index=False)
#df2.to_csv('feature_vector_1sy8.csv', index=False)

feature_names = ['rad', 'CN', 'q', 'hbond'] + list(df3.columns[df3.columns.str.startswith('name') | df3.columns.str.startswith('resname') | df3.columns.str.startswith('neighbor_resnames_0_') | df3.columns.str.startswith('neighbor_resnames_1_') | df3.columns.str.startswith('neighbor_names_0_') | df3.columns.str.startswith('neighbor_names_1_')]) 

feature_names_test = ['rad', 'CN', 'q', 'hbond'] + list(df2.columns[df2.columns.str.startswith('name') | df2.columns.str.startswith('resname') | df2.columns.str.startswith('neighbor_resnames_0_') | df2.columns.str.startswith('neighbor_resnames_1_') | df2.columns.str.startswith('neighbor_names_0_') | df2.columns.str.startswith('neighbor_names_1_')])

chem_shift = ['chem_shift']

# Split the data into features and labels
X = df3[feature_names].values
#np.savetxt('X_H.csv', X, delimiter=',', header=','.join(feature_names), comments='')
y = df3[chem_shift].values.ravel()
#np.savetxt('y_H.csv', y, delimiter=',', header=','.join(feature_names), comments='')

X_test = df2[feature_names_test].values
np.savetxt('X_test.csv', X_test, delimiter=',', header=','.join(feature_names), comments='')
#y_exp_cs = df2[chem_shift].values.ravel()
#np.savetxt('y_exp_cs.csv', y_exp_cs, delimiter=',', header=','.join(feature_names), comments='')
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest Regressor
regr = RandomForestRegressor(n_estimators=500, random_state=0, max_depth=30,
                              min_samples_split=2,
                              min_samples_leaf=1)

regr.fit(X_scaled, y)
# Predict the chemical shifts on the training set
y_pred = regr.predict(X_scaled)
# calculate R2 score between y_test and y_pred
r2 = r2_score(y, y_pred)
print('R2 score:', r2)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)

# Predict the chemical shifts on the test molecule
#y1_pred = regr.predict(X_train)
#np.savetxt('y_pred', y_pred, delimiter=',', header=','.join(feature_names), comments='')

X_test_scaled = scaler.fit_transform(X_test)
y_test = regr.predict(X_test_scaled)
np.savetxt('y_test_pred.csv', y_test, delimiter=',', header=','.join(feature_names), comments='')
#r22 = r2_score(y_exp_cs, y_pred_exp)
#mae2 = mean_absolute_error(y_exp_cs, y_pred_exp)
#print('R2 score_exp:', r22)
#print("Mean Absolute Error_exp:", mae2)


###MAD###
# Use polyfit to perform a linear fit
coefficients = np.polyfit(y, y_pred, 1)
# The coefficients returned by polyfit are the intercept and slope of the line
intercept = coefficients[1]
slope = coefficients[0]

ynew = []
diff = []
for x, y  in zip(y, y_pred):
        diff.append(abs((((y - intercept) / slope )) - x ))
        ynew.append((slope * x) + intercept)
        total = sum(diff)
        length = len(diff)
        mad = total / length
print("MAD:", mad)

