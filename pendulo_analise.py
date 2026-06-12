# APS Pêndulo Simples — Física Experimental
# Rastreia a laranja frame a frame e ajusta o OHA nos dados

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os, sys, warnings
warnings.filterwarnings("ignore")

COMPRIMENTO_FIO_CM  = 46.0
ERRO_COMPRIMENTO_CM = 0.23  # incerteza total relacionada com a medida do cadarço pela fita métrica
MASSA_G             = 189.0
ERRO_MASSA_G        = 1.0   # incerteza da balança 
GRAVIDADE           = 9.81

# faixa de cor laranja no HSV 
COR_BAIXO_HSV = np.array([ 8, 140,  80], dtype=np.uint8)
COR_ALTO_HSV  = np.array([25, 255, 255], dtype=np.uint8)

AREA_MINIMA_PX  = 800   
KERNEL_EROSAO   = 5
KERNEL_DILATACAO= 9

ARQUIVO_VIDEO   = "pendulo.mp4"
ARQUIVO_CSV     = "posicoes_pendulo.csv"
ARQUIVO_GRAFICO = "analise_pendulo.png"


def extrair_posicoes(caminho_video):
    captura = cv2.VideoCapture(caminho_video)
    if not captura.isOpened():
        print("Erro: não consegui abrir o vídeo.")
        sys.exit(1)

    fps          = captura.get(cv2.CAP_PROP_FPS)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    largura      = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura       = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps:.1f} | Resolução: {largura}x{altura} | Duração: {total_frames/fps:.1f}s")

    # pula os primeiros 10 segundos (pêndulo ainda sendo solto)
    frame_inicial = int(fps * 10)
    captura.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)

    kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_EROSAO, KERNEL_EROSAO))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_DILATACAO, KERNEL_DILATACAO))

    registros    = []
    numero_frame = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        tempo_s    = (frame_inicial + numero_frame) / fps
        frame_suav = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv        = cv2.cvtColor(frame_suav, cv2.COLOR_BGR2HSV)
        mascara    = cv2.inRange(hsv, COR_BAIXO_HSV, COR_ALTO_HSV)

        # corta região onde aparecem objetos que atrapalham
        h, w = frame.shape[:2]
        mascara[:int(h * 0.4), :] = 0   # ignora parte de cima
        mascara[:, int(w * 0.75):] = 0  # ignora parte direita

        mascara = cv2.erode(mascara,  kernel_ero, iterations=1)
        mascara = cv2.dilate(mascara, kernel_dil, iterations=2)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_det, y_det = np.nan, np.nan

        if contornos:
            # filtra por área e circularidade para pegar só a laranja
            candidatos = []
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area < AREA_MINIMA_PX:
                    continue
                perim = cv2.arcLength(cnt, True)
                if perim == 0:
                    continue
                circularidade = 4 * np.pi * area / (perim ** 2)
                if circularidade > 0.5:
                    candidatos.append((area, cnt))

            if candidatos:
                _, maior = max(candidatos, key=lambda c: c[0])
                M = cv2.moments(maior)
                if M["m00"] > 0:
                    x_det = int(M["m10"] / M["m00"])
                    y_det = int(M["m01"] / M["m00"])

        registros.append({"frame": frame_inicial + numero_frame,
                          "tempo_s": round(tempo_s, 6),
                          "x_px": x_det, "y_px": y_det})
        numero_frame += 1

        if numero_frame % int(fps * 5) == 0:
            print(f"  {100*numero_frame/(total_frames-frame_inicial):.1f}% processado...")

    captura.release()
    df = pd.DataFrame(registros)
    df.to_csv(ARQUIVO_CSV, index=False)
    print(f"Posições salvas em {ARQUIVO_CSV}")
    return df


def preprocessar_dados(df):
    df = df.dropna(subset=["x_px", "y_px"]).copy()
    df.reset_index(drop=True, inplace=True)

    # suavização simples pra reduzir ruído de rastreamento
    df["x_suav"] = df["x_px"].rolling(window=5, center=True, min_periods=1).median()

    x_eq = df["x_suav"].mean()
    df["x_centrado"] = df["x_suav"] - x_eq

    print(f"Equilíbrio em {x_eq:.1f} px | {len(df)} frames válidos")
    return df


def oha(t, A, b, omega, phi, x0):
    # equação do oscilador harmônico amortecido
    return A * np.exp(-b * t) * np.cos(omega * t + phi) + x0


def ajustar_oha(df):
    t = df["tempo_s"].values
    x = df["x_centrado"].values

    A_ini = (x.max() - x.min()) / 2.0

    picos, _ = find_peaks(x, height=A_ini * 0.3)
    if len(picos) >= 2:
        T_ini     = np.median(np.diff(t[picos]))
        omega_ini = 2 * np.pi / T_ini
    else:
        # fallback pelo comprimento do fio
        omega_ini = np.sqrt(GRAVIDADE / (COMPRIMENTO_FIO_CM / 100))

    p0      = [A_ini, 0.05, omega_ini, 0.0, 0.0]
    limites = ([0, 0, 0.1, -np.pi, -200],
               [5000, 5.0, 100, np.pi, 200])

    try:
        params, cov = curve_fit(oha, t, x, p0=p0, bounds=limites, maxfev=20000, method="trf")
    except RuntimeError:
        print("Ajuste não convergiu, usando estimativas iniciais.")
        params = np.array(p0)
        cov    = np.diag([1e6] * 5)

    erros = np.sqrt(np.diag(cov))
    A, b, omega, phi, x0_fit = params

    L_m = COMPRIMENTO_FIO_CM / 100.0
    x_aj = oha(t, *params)

    return {
        "A": A, "dA": erros[0],
        "b": b, "db": erros[1],
        "omega": omega, "domega": erros[2],
        "phi": phi, "dphi": erros[3],
        "x0": x0_fit, "dx0": erros[4],
        "periodo_exp":     2 * np.pi / omega,
        "frequencia":      omega / (2 * np.pi),
        "tau":             1.0 / b if b > 0 else np.inf,
        "fator_qualidade": omega / (2 * b) if b > 0 else np.inf,
        "periodo_teo":     2 * np.pi * np.sqrt(L_m / GRAVIDADE),
        "erro_periodo_teo": np.pi * np.sqrt(1 / (GRAVIDADE * L_m)) * (ERRO_COMPRIMENTO_CM / 100),
        "residuo_rms":     np.sqrt(np.mean((x - x_aj) ** 2)),
        "t": t, "x": x, "x_ajustado": x_aj,
    }


def imprimir_relatorio(res):
    print("\n--- RESULTADOS ---")
    print(f"m  = {MASSA_G:.1f} ± {ERRO_MASSA_G:.1f} g")
    print(f"A  = {res['A']:.2f} ± {res['dA']:.2f} px")
    print(f"b  = {res['b']:.5f} ± {res['db']:.5f} s⁻¹")
    print(f"ω  = {res['omega']:.4f} ± {res['domega']:.4f} rad/s")
    print(f"φ  = {res['phi']:.4f} ± {res['dphi']:.4f} rad")
    print(f"T_exp = {res['periodo_exp']:.4f} s")
    print(f"T_teo = {res['periodo_teo']:.4f} ± {res['erro_periodo_teo']:.4f} s")
    print(f"τ  = {res['tau']:.2f} s")
    print(f"Q  = {res['fator_qualidade']:.2f}")
    print(f"RMS = {res['residuo_rms']:.2f} px")
    disc = abs(res['periodo_exp'] - res['periodo_teo'])
    print(f"|T_exp - T_teo| = {disc:.4f} s")


def plotar_resultados(df, res):
    t, x, x_aj = res["t"], res["x"], res["x_ajustado"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Análise do Pêndulo Simples — OHA", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(t, x, s=3, color="steelblue", alpha=0.5, label="Dados experimentais")
    ax1.plot(t, x_aj, color="crimson", linewidth=1.8, label="Ajuste OHA")
    ax1.plot(t,  res["A"] * np.exp(-res["b"] * t), "darkorange", linewidth=1.2, linestyle="--", label="Envelope")
    ax1.plot(t, -res["A"] * np.exp(-res["b"] * t), "darkorange", linewidth=1.2, linestyle="--")
    ax1.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Posição x (px, centrada)")
    ax1.set_title(f"x(t) = {res['A']:.1f}·exp(−{res['b']:.4f}·t)·cos({res['omega']:.4f}·t + {res['phi']:.3f})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    residuos = x - x_aj
    ax2 = fig.add_subplot(gs[1, :])
    ax2.scatter(t, residuos, s=3, color="seagreen", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.fill_between(t, -res["residuo_rms"], res["residuo_rms"], alpha=0.15, color="seagreen",
                     label=f"±RMS = ±{res['residuo_rms']:.1f} px")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Resíduo (px)")
    ax2.set_title("Resíduos do ajuste")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    dados_tabela = [
        ["Parâmetro", "Valor", "Incerteza"],
        ["m (g)", f"{MASSA_G:.1f}", f"±{ERRO_MASSA_G:.1f}"],
        ["A (px)",    f"{res['A']:.2f}",          f"±{res['dA']:.2f}"],
        ["b (s⁻¹)",  f"{res['b']:.5f}",          f"±{res['db']:.5f}"],
        ["ω (rad/s)", f"{res['omega']:.4f}",      f"±{res['domega']:.4f}"],
        ["φ (rad)",   f"{res['phi']:.4f}",        f"±{res['dphi']:.4f}"],
        ["T_exp (s)", f"{res['periodo_exp']:.4f}", "—"],
        ["T_teo (s)", f"{res['periodo_teo']:.4f}", f"±{res['erro_periodo_teo']:.4f}"],
        ["τ (s)",     f"{res['tau']:.2f}",         "—"],
        ["Q",         f"{res['fator_qualidade']:.2f}", "—"],
        ["RMS (px)",  f"{res['residuo_rms']:.2f}", "—"],
    ]
    tabela = ax3.table(cellText=dados_tabela[1:], colLabels=dados_tabela[0],
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    for (lin, col), cel in tabela.get_celld().items():
        if lin == 0:
            cel.set_facecolor("#2c3e50")
            cel.set_text_props(color="white", fontweight="bold")
        elif lin % 2 == 0:
            cel.set_facecolor("#ecf0f1")
    ax3.set_title("Resumo dos parâmetros", fontsize=10)

    ax4 = fig.add_subplot(gs[2, 1])
    sc = ax4.scatter(df["x_px"], df["y_px"], s=2, c=df["tempo_s"], cmap="plasma", alpha=0.6)
    fig.colorbar(sc, ax=ax4, label="Tempo (s)")
    ax4.set_xlabel("x (px)")
    ax4.set_ylabel("y (px)")
    ax4.set_title("Trajetória da massa")
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)

    plt.savefig(ARQUIVO_GRAFICO, dpi=150, bbox_inches="tight")
    plt.show()


def testar_rastreamento(caminho_video):
    captura = cv2.VideoCapture(caminho_video)
    if not captura.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    print("Modo teste — pressione 'q' pra sair, 's' pra salvar frame")
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_DILATACAO, KERNEL_DILATACAO))

    while True:
        ret, frame = captura.read()
        if not ret:
            captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, COR_BAIXO_HSV, COR_ALTO_HSV)

        h, w = frame.shape[:2]
        mascara[:int(h * 0.4), :] = 0
        mascara[:, int(w * 0.75):] = 0

        mascara = cv2.dilate(mascara, kernel_dil, iterations=2)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < AREA_MINIMA_PX:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            if 4 * np.pi * area / (perim ** 2) > 0.5:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(vis, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.drawContours(vis, [cnt], -1, (0, 255, 255), 2)

        saida = np.hstack([cv2.resize(vis, (640, 360)),
                           cv2.resize(cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR), (640, 360))])
        cv2.imshow("original | mascara  ('q' sai)", saida)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("s"):
            cv2.imwrite("frame_teste.png", saida)
            print("Frame salvo.")

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Pêndulo — L={COMPRIMENTO_FIO_CM}cm | m={MASSA_G}g")

    MODO = "teste"  # trocar para "teste" para ver o decalque

    if MODO == "teste":
        testar_rastreamento(ARQUIVO_VIDEO)
    else:
        if os.path.exists(ARQUIVO_CSV):
            resp = input(f"'{ARQUIVO_CSV}' já existe. Usar dados salvos? (s/n): ").strip().lower()
            df_bruto = pd.read_csv(ARQUIVO_CSV) if resp == "s" else extrair_posicoes(ARQUIVO_VIDEO)
        else:
            df_bruto = extrair_posicoes(ARQUIVO_VIDEO)

        df_limpo  = preprocessar_dados(df_bruto)
        resultado = ajustar_oha(df_limpo)
        imprimir_relatorio(resultado)
        plotar_resultados(df_limpo, resultado)