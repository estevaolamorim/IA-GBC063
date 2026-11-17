"""Gráficos dos resultados (matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def media_movel(x, janela=50):
    x = np.asarray(x, dtype=float)
    if len(x) < janela:
        janela = max(1, len(x))
    return np.convolve(x, np.ones(janela) / janela, mode="valid")


def curva_aprendizado(retornos, referencia, caminho):
    """Lucro por episódio durante o treino, com a política de referência."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(retornos, color="#9fb3d9", alpha=0.5, label="Episódio")
    mm = media_movel(retornos, 50)
    plt.plot(np.arange(len(mm)) + 49, mm, color="#14213d", lw=2.5, label="Média móvel (50)")
    plt.axhline(referencia, color="#b44f0a", ls="--", lw=2, label="Reposição até 25 (referência)")
    plt.xlabel("Episódio de treino")
    plt.ylabel("Lucro em 30 dias")
    plt.title("Curva de aprendizado do DQN")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def comparacao_politicas(resultados, caminho):
    """Barras com lucro médio de cada política (barra de erro = erro padrão)."""
    nomes = list(resultados)
    medias = [resultados[n]["media"] for n in nomes]
    erros = [resultados[n]["erro_padrao"] for n in nomes]
    cores = ["#14213d" if n == "DQN" else "#9fb3d9" for n in nomes]
    plt.figure(figsize=(8, 4.5))
    plt.bar(nomes, medias, yerr=erros, color=cores, capsize=6)
    plt.ylabel("Lucro médio em 30 dias")
    plt.title("DQN x políticas de referência")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()


def episodio_exemplo(registro, caminho):
    """Um episódio dia a dia: estoque, pedidos e demanda."""
    dias = np.arange(1, len(registro) + 1)
    estoque = [d["estoque_fim"] for d in registro]
    pedidos = [d["pedido"] for d in registro]
    demanda = [d["demanda"] for d in registro]
    plt.figure(figsize=(9, 4.5))
    plt.bar(dias, pedidos, color="#f4a259", alpha=0.8, label="Pedido (unidades)")
    plt.plot(dias, estoque, color="#14213d", lw=2.5, marker="o", label="Estoque no fim do dia")
    plt.plot(dias, demanda, color="#1d7a6f", lw=1.5, ls="--", marker="x", label="Demanda do dia")
    plt.xlabel("Dia do episódio")
    plt.ylabel("Unidades")
    plt.title("Exemplo de episódio com o DQN treinado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
