import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("dados_vendas.csv")



while True:
    print("=" * 25)
    opcao = int(input(
        "[1] Visualizar gráfico de previsão\n"
        "[2] Realizar teste de previsão\n"
        "[3] Sair do programa\n"
        "Selecione uma opcao:"
    ))

    if opcao == 1:
        sb.set_theme(style="whitegrid", palette="viridis")
        sb.scatterplot(df, x="investimento", y="vendas", label="Valores reais")
        plt.plot(df[["investimento"]],df[["vendas"]])
        plt.title("Previsão de renda com base nas vendas", fontweight="bold")
        plt.show()

    elif opcao == 2:
        modelo = sk.linear_model.LinearRegression()
        modelo.fit(df[["investimento"]], df[["vendas"]])

        investido = int(input("Digite o valor investido: "))
        venda = round(modelo.predict([[investido]])[0][0])
        print(F"Número de vendas estimado: {venda}")
    elif opcao == 3:
        print("=" * 25)
        break



