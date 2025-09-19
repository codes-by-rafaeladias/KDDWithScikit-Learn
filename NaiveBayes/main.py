import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

#caminho do arquivo de treinamento 
TRAIN = "NaiveBayes\\files\\maupagador_train.xlsx"
TEST = {
    "Casa Própria": ["Não"],
    "EstCivil": ["Divorciado"],
    "Rendim.": ["médio"],
    "Mau Pagador": [""]
}

def main(test, train):
    df_train = pandas.read_excel(train)
    print("Treinamento:", df_train.head()) 
    df_test = pandas.DataFrame(test)
    
    #1° passo - pré-processamento
    
    X_train = df_train.drop(columns=["Mau Pagador"]) 
    y_train = df_train["Mau Pagador"]  
    
    #preparar os dados do treinamento
    
    encoders = {}
    for column in X_train.columns:
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])
        encoders[column] = le 
        
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    
    #preparar os dados do teste
    X_test = df_test.drop(columns=["Mau Pagador"])
    for column in X_test.columns:
        le = encoders[column] 
        X_test[column] = le.transform(X_test[column])
        
    #criar e treinar o modelo
    model = CategoricalNB()
    model.fit(X_train, y_train)
    
    #fazer as previsões com base nas regras dadas pelo modelo
    y_pred = model.predict(X_test)
    
    df_test["Mau Pagador"] = le_target.inverse_transform(y_pred)
    
    #previsão realizada é impressa
    print("Previsão: ")
    print(df_test)

if __name__ == "__main__":
    main(TEST, TRAIN)