import numpy as np
import pandas as pd
import adaline as ad

model = None
_trained = False

df = pd.read_csv('train.csv')
df_cleaned = df.drop(columns = ['Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'])

missing_age_indices = df_cleaned['Age'].isnull()
random_age = np.random.randint(1, 65, size = missing_age_indices.sum())
df_cleaned.loc[missing_age_indices, 'Age'] = random_age

df_cleaned['Sex'] = df_cleaned['Sex'].map({'male': 1, 'female': 0})

X = df_cleaned[['Pclass', 'Age', 'Sex', 'Parch', 'SibSp']].values
y = df_cleaned['Survived'].values

X_std = np.copy(X)
for i in range(X.shape[1]):
    X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()    #X_std = (X - X.mean) / X.std  

while True:
    print("-----------------------------------------")
    print("🚢 Titanic Survival Prediction")
    print("Premi 1 per addestrare")
    print("Premi 2 per testare con dati di test")
    print("Premi 3 per testare con dati da inserire")
    print("Premi 4 per uscire")
    print("-----------------------------------------")
    risposta = input()
    if risposta == '1':
        model = ad.AdalineSGD(eta=0.01, n_iter=100, random_state=1)
        model.fit(X_std, y)
        _trained = True
        print("✅ Addestramento completato")
        for i, c in enumerate(model.cost_, start=1):
            print(f"Epoca {i}: costo medio = {c:.4f}")
    elif risposta == '2' and _trained == True:
        if model is None:
            print("⚠️ Il modello non è stato ancora addestrato. \n")
        else:
            df_test = pd.read_csv('test.csv')
            df_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'])

            missing_age_indices = df_test['Age'].isnull()
            random_age = np.random.randint(1, 65, size=missing_age_indices.sum())
            df_test.loc[missing_age_indices, 'Age'] = random_age
            df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})

            X_test = df_test[['Pclass', 'Age', 'Sex', 'Parch', 'SibSp']].values
            for i in range(X_test.shape[1]):
                X_test[:, i] = (X_test[:, i] - X[:, i].mean()) / X[:, i].std()  # standardizziamo con media e std del training

            y_pred = model.predict(X_test)
            print("✅ Predizione completata. Ecco i primi 10 risultati:")
            print(y_pred[:10])
    elif risposta == '3' and _trained == True:
        print("Inserisci i dati del passeggero (Pclass, Age, Sex (1=male, 0=female), Parch, SibSp):")
        valori = input("Separati da virgola: ")
        dati = np.array([float(x) for x in valori.split(',')])
        dati_std = (dati - X.mean(axis=0)) / X.std(axis=0)
        predizione = model.predict(dati_std.reshape(1, -1))
        risultato = "Sopravvissuto" if predizione[0] == 1 else "Non sopravvissuto"
        print(f"🔍 Risultato della predizione: {risultato}")
    elif risposta == '4':
        break
    else: 
        print("⚠️ Scelta non valida, modello non addestrato. \n")

