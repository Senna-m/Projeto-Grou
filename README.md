# 🚐 Assistente Tráfego

Sistema de recomendação de veículos para logística de turismo, baseado em análise preditiva de consumo de combustível.

---

## 🎯 Problema resolvido

Empresas de transporte turístico precisam escolher qual veículo usar em cada rota. Essa decisão impacta diretamente o custo com combustível. Este sistema analisa o histórico real de abastecimento e rotas anteriores para **recomendar automaticamente o veículo mais eficiente** para uma nova rota.

---

## 🛠️ Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| ML / Modelo | CatBoost (`.cbm`) |
| Análise de dados | Pandas, Scikit-learn |
| Visualização | Matplotlib |
| Backend / API | FastAPI / Flask + Uvicorn |
| Frontend / Deploy | Streamlit |
| Treinamento | Google Colab |

---

## ⚙️ Como funciona

```
Usuário informa a rota
        ↓
Sistema carrega histórico de consumo e rotas anteriores
        ↓
Modelo CatBoost calcula previsão de consumo por veículo
        ↓
Sistema retorna o veículo mais eficiente para aquela rota
```

---

## 📊 Funcionalidades

- ✅ Análise e limpeza de dados históricos de abastecimento
- ✅ Treinamento de modelo preditivo com CatBoost
- ✅ API para consulta de recomendação de veículo
- ✅ Interface web interativa com Streamlit
- ✅ Exportação e carregamento de modelo treinado (`.cbm`)
- ✅ Redução de custos via otimização de consumo

---

## 📂 Estrutura do projeto

```
Projeto-Grou/
├── app.py                  # Aplicação Streamlit + lógica de recomendação
├── modelo_diesel.cbm       # Modelo CatBoost treinado e exportado
├── requirements.txt        # Dependências do projeto
├── data/                   # Dados históricos de abastecimento e rotas
├── projeto/                # Notebooks de análise e treinamento
│   └── *.ipynb             # EDA, feature engineering, treinamento
└── catboost_info/          # Logs de treinamento do modelo
```

---

## 🚀 Como rodar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/Senna-m/Projeto-Grou.git
cd Projeto-Grou

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Rode a aplicação
streamlit run app.py
```

---

## 🧠 Decisões técnicas

- **CatBoost** foi escolhido por lidar bem com variáveis categóricas (tipo de veículo, rota) sem necessidade de encoding manual.
- O modelo foi treinado no Google Colab e exportado no formato `.cbm` para uso em produção.
- A interface em Streamlit permite uso imediato sem necessidade de front-end separado.

---

## 👩‍💻 Sobre

Desenvolvido por **Nathalie Senna** — [GitHub](https://github.com/Senna-m) · [LinkedIn](https://linkedin.com/in/nathalie-senna-a37b483b0/)
