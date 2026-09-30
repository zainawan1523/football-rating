import glob
import os
import pickle

import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Features: order wahi rakho jo app.py me hai
FEATURES = ['age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
TARGET = 'overall'
COLUMNS = ['fifa_version'] + FEATURES + [TARGET]

# 1. male_players file dhoondo (.csv, .xlsx ya .xls, jo bhi is folder me ho)
found = sorted(glob.glob('male_players*'), key=lambda p: (not p.lower().endswith('.csv'), p))
found = [p for p in found if p.lower().endswith(('.csv', '.xlsx', '.xls'))]
if not found:
    raise SystemExit("male_players file is folder me nahi mili. 'dir' chala kar check karo.")

path = found[0]
ext = os.path.splitext(path)[1].lower()
print('File use ho rahi hai:', path)

# 2. File load karo (sirf zaroori columns)
if ext == '.csv':
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
else:
    df = pd.read_excel(path, usecols=COLUMNS)
    if ext == '.xls':
        print("WARNING: .xls me sirf 65,536 rows aati hain, data kata hua ho sakta hai. "
              "Original .csv istemal karna behtar hai.")

print('Rows load hui:', len(df))
print('FIFA versions file me:', sorted(df['fifa_version'].dropna().unique().tolist()))

# 3. Latest version lo (FC 24 = 24). Agar 24 nahi mila to jo latest hai wo lo
latest = df['fifa_version'].max()
if latest != 24:
    print(f'WARNING: FC 24 (version 24) file me nahi mila, version {latest:.0f} use ho raha hai.')
df = df[df['fifa_version'] == latest]
print('Players is version me:', len(df))

# 4. Goalkeepers ke pace/shooting waghera NaN hote hain, unko hata do
df = df.dropna(subset=FEATURES + [TARGET])
print('Outfield players (NaN hatane ke baad):', len(df))

X = df[FEATURES].values
y = df[TARGET].values

# 5. Train / test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model train karo
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Model test karo
pred = model.predict(X_test)
print('\nR2 score :', round(r2_score(y_test, pred), 4))
print('MAE      :', round(mean_absolute_error(y_test, pred), 2), '(overall points)')
print('Coef     :', {f: round(float(c), 3) for f, c in zip(FEATURES, model.coef_)})
print('Intercept:', round(float(model.intercept_), 3))

# 8. model.pkl me save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('\nmodel.pkl save ho gaya.')
print('requirements.txt me yeh likhna: scikit-learn==' + sklearn.__version__)