import pandas as pd
df=pd.read_csv("Abastecimento.csv")
df = df.drop(columns=['DESCONTOS','1','2','3','4','5','6','7','8','9','10'])
df.to_csv("arquivo_limpo.csv", index=False)

print("Arquivo limpo com sucesso!")

