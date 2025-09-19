import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#caminhos dos arquivos de treinamento e teste
TRAIN = "DecisionTreeClassifier\\files\\lentes_treinamento.xlsx"
TEST = "DecisionTreeClassifier\\files\\lentes_teste.xlsx"

def main(test, train):
    df_train = pandas.read_excel(train)
    print("Treinamento:", df_train.head()) 
    df_test = pandas.read_excel(test)
    print("Teste:", df_test.head()) 
    
    #1° passo - pré-processamento
    
    X_train = df_train.drop(columns=["LENTES"]) 
    y_train = df_train["LENTES"]  
    
    #preparar os dados do treinamento
    
    encoders = {}
    for column in X_train.columns:
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])
        encoders[column] = le 
        
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    
    #preparar os dados do teste
    X_test = df_test.drop(columns=["LENTES", "PREDIÇÃO"])
    for column in X_test.columns:
        le = encoders[column] 
        X_test[column] = le.transform(X_test[column])
    
    y_test = le_target.transform(df_test["LENTES"])
        
    #criar e treinar o modelo
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    #fazer as previsões com base nas regras dadas pelo modelo
    y_pred = model.predict(X_test)
    
    df_test["PREDIÇÃO"] = le_target.inverse_transform(y_pred)
    
    #avaliar o modelo
    acuraccy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {acuraccy:.2f}")

if __name__ == "__main__":
    main(TEST, TRAIN)