import pandas as pd
df=pd.read_csv("Abastecimento.csv")
df.loc[df["ANO"].isna() | (df["ANO"] == " "), "ANO"] = "2022.0"
df.to_csv("Abastecimento.csv", index=False)

print("Arquivo limpo com sucesso!")

