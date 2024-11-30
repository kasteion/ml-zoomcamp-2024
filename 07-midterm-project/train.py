import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

# Feature lists
category_cols = ['client_catg', 'region', 'district']
numeric_cols = ['age_months', 'consumption_level_1_mean',
    'consumption_level_1_std', 'consumption_level_1_min',
    'consumption_level_1_max', 'consumption_level_2_mean',
    'consumption_level_2_std', 'consumption_level_2_min',
    'consumption_level_2_max', 'consumption_level_3_mean',
    'consumption_level_3_std', 'consumption_level_3_min',
    'consumption_level_3_max', 'consumption_level_4_mean',
    'consumption_level_4_std', 'consumption_level_4_min',
    'consumption_level_4_max', 'total_consumption_mean',
    'total_consumption_std', 'total_consumption_min',
    'total_consumption_max']

# Model parameters
max_depth=20
min_samples_leaf=5

# Output file
output_dv = 'dv.bin'
output_model = f'model_d={max_depth}s={min_samples_leaf}.bin'

# Load data
df_clients = pd.concat([pd.read_csv('data/client_train.csv')])
df_invoices = pd.concat([pd.read_csv('data/invoice_train.csv')])

# Editing clients dataset
df_clients['district'] = df_clients['disrict']
del df_clients['disrict']
df_clients.creation_date = pd.to_datetime(df_clients.creation_date, dayfirst=True)
df_clients['client_id'] = df_clients['client_id'].str.lower()
df_clients['district'] = 'district_' + df_clients['district'].astype('str')
df_clients['region'] = 'region_' + df_clients['region'].astype('str')
df_clients['client_catg'] = 'client_catg_' + df_clients['client_catg'].astype('str')
df_clients['target'] = df_clients['target'].astype(int)

# Editing invoices dataset
df_invoices['client_id'] = df_invoices['client_id'].str.lower()
df_invoices['counter_type'] = df_invoices['counter_type'].str.lower()
df_invoices['counter_status'] = df_invoices['counter_statue']
df_invoices['consumption_level_1'] = df_invoices['consommation_level_1']
df_invoices['consumption_level_2'] = df_invoices['consommation_level_2']
df_invoices['consumption_level_3'] = df_invoices['consommation_level_3']
df_invoices['consumption_level_4'] = df_invoices['consommation_level_4']
del df_invoices['counter_statue']
del df_invoices['consommation_level_1']
del df_invoices['consommation_level_2']
del df_invoices['consommation_level_3']
del df_invoices['consommation_level_4']
df_invoices.counter_status = pd.to_numeric(df_invoices.counter_status, errors='coerce').fillna(999999).astype(int)
df_invoices = df_invoices[df_invoices.counter_status <= 5]
df_invoices['counter_status'] = 'counter_status_' + df_invoices['counter_status'].astype(str)
df_invoices.invoice_date = pd.to_datetime(df_invoices.invoice_date, yearfirst=True)

# Extending dataset
max_date = df_invoices.invoice_date.max()
df_clients_extended = df_clients.copy()
df_invoices_extended = df_invoices.copy()
df_invoices_extended = df_invoices_extended.merge(df_clients[['client_id', 'target']], on='client_id', how='left')
df_clients_extended['age_months'] = (max_date.year - df_clients_extended['creation_date'].dt.year) * 12 + (max_date.month-  df_clients_extended['creation_date'].dt.month)
df_invoices_extended['total_consumption'] = (df_invoices_extended['consumption_level_1'] + df_invoices_extended['consumption_level_2'] + df_invoices_extended['consumption_level_3'] + df_invoices_extended['consumption_level_4'])
consumption_cols = ['consumption_level_1', 'consumption_level_2', 'consumption_level_3', 'consumption_level_4', 'total_consumption']
for c in consumption_cols:
    df_agg = df_invoices_extended.groupby('client_id')[c].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_agg.rename(columns={'mean': f'{c}_mean', 'std': f'{c}_std', 'min': f'{c}_min', 'max': f'{c}_max'}, inplace=True)
    df_clients_extended = df_clients_extended.merge(df_agg, on='client_id', how='left')
df_clients_extended = df_clients_extended.fillna(0)

# Spliting dataset
df_clients_full_train, df_clients_test = train_test_split(df_clients_extended, test_size=0.2, random_state=42)
df_clients_full_train.reset_index(drop=True)
df_clients_test.reset_index(drop=True)
y_clients_full_train = df_clients_full_train.target
y_clients_test = df_clients_test.target
del df_clients_full_train['target']
del df_clients_test['target']

# Training model
dicts = df_clients_full_train[category_cols + numeric_cols].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dicts)

dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
dt.fit(X_train, y_clients_full_train)

with open(output_dv, 'wb') as file:
    pickle.dump(dv, file)

with open(output_model, 'wb') as file:
    pickle.dump(dt, file)