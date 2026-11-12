"""Gera notebooks/controles_E9_E10.ipynb: controles pedidos no parecer do benchmark.

E9  contexto com as mesmas janelas e o mesmo número de atualizações (só o contexto muda)
E10 benchmark com agrupamento espacial rigoroso (ligação simples, r >= 0,85)
"""
import base64, io, json, os, zipfile, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mk", os.path.join(ROOT, "make_notebook.py"))
src = open(os.path.join(ROOT, "make_notebook.py"), encoding="utf8").read()
# reaproveita PKG, code, md e UI do gerador principal sem executar a montagem do notebook dele
head = src.split("cells = [")[0]
ns = {"__file__": os.path.join(ROOT, "make_notebook.py")}
exec(head, ns)
PKG, code, md, UI = ns["PKG"], ns["code"], ns["md"], ns["UI"]

cells = [
    md("""# Controles do benchmark (resposta ao parecer)
**E9 · contexto com janelas fixas.** Em cada fold, as janelas de treino, validação e teste são definidas uma vez com 360 meses
de contexto. Para L = 24, 60, 120, 240 e 360 cada modelo lê só os últimos L meses dessas mesmas janelas: mesmos pares
(bacia, origem), mesmos alvos, mesmo número de janelas, mesma sequência de minibatches e 3.000 atualizações. Só o contexto muda.
Chronos-Bolt-small e Chronos-2 recebem exatamente os mesmos contextos (e o histórico completo, até 512 meses).

**E10 · agrupamento rigoroso.** Grupos por ligação simples em 1 − r com corte 0,15: nenhuma bacia de teste tem vizinho no pool
com correlação de anomalias diárias ≥ 0,85 (52 grupos; o maior reúne as 27 bacias do Grande, Paranaíba e alto Paraná).
Os modelos treinados são refeitos com os novos folds, nos dois cenários; PAR(p) e Chronos zero-shot também, para os pareados.

**Use GPU.** Tudo fica salvo no Drive; se a sessão cair, rode de novo e ele continua. Tempo estimado numa L4/A100:
E9 com 5 sementes ≈ 6–8 h; E10 ≈ 3–5 h. Com 3 sementes cai para ~60%."""),
    code("""
!pip -q install "chronos-forecasting>=2.2" lightgbm peft pyarrow openpyxl
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA (ative a GPU)")""", "1 · Instalar dependências"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
ARQUIVO_VAZOES = "/content/drive/MyDrive/reservatorios/Vazões_Diárias_1931_2024.xlsx"  #@param {type:"string"}
PASTA_RESULTADOS = "/content/drive/MyDrive/reservatorios/resultados_v2"  #@param {type:"string"}
PASTA_RIGOROSO = "/content/drive/MyDrive/reservatorios/resultados_v2_rigoroso"  #@param {type:"string"}""",
         "2 · Arquivos (use a MESMA pasta de resultados do benchmark)"),
    code(f"""
import base64, io, zipfile, sys, os
_PKG = "{PKG}"
os.makedirs("/content/hb", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(_PKG))).extractall("/content/hb")
sys.path.insert(0, "/content/hb")
for m in [k for k in sys.modules if k.startswith("hydrobench")]:
    del sys.modules[m]
import hydrobench as HB
from hydrobench import analysis as A, controls as K
{UI}
painel("Pacote hydrobench carregado", "<p class='s'>inclui hydrobench.controls (E9, E10)</p>")""", "3 · Pacote do estudo (oculto)"),
    code("""
N_SEMENTES = 5  #@param {type:"integer"}
RODAR_E9 = True  #@param {type:"boolean"}
RODAR_E10 = True  #@param {type:"boolean"}
cfg = HB.Config(data_path=ARQUIVO_VAZOES, out_dir=PASTA_RESULTADOS, seeds=list(range(N_SEMENTES)))
mon = Monitor("Preparando dados (estudo principal)")
st = HB.Study(cfg, log=mon)
kpis([("bacias", len(st.basins)), ("grupos", st.groups.nunique()), ("folds", cfg.n_folds)], "Estudo principal")
rows = []
for f in range(cfg.n_folds):
    tr, va, _, _ = st.pool("spatiotemporal", f, K.E9_REF)
    t = st.test(f, K.E9_REF)
    rows.append(dict(fold=f, janelas_treino=len(tr["x"]), janelas_validacao=len(va["x"]), janelas_teste=len(t["x"])))
tabela(_pd.DataFrame(rows), "E9 · janelas fixas por fold (idênticas para todos os L)")""", "4 · Preparar"),
    code("""
if RODAR_E9:
    mon = Monitor("E9 · contexto com janelas e atualizações fixas")
    st.log = mon
    K.run_e9(st)
    kpis([("execuções E9 salvas", contar(st, "E9_"))], "E9 concluído")""", "5 · E9 (GPU)"),
    code("""
if RODAR_E10:
    mon = Monitor("E10 · agrupamento rigoroso")
    st.log = mon
    csv = os.path.join(PASTA_RESULTADOS, "tables", "grupos_rigorosos_single_085.csv")
    g, Cm = K.write_strict_csv(st, csv)
    cfg2 = HB.Config(data_path=ARQUIVO_VAZOES, out_dir=PASTA_RIGOROSO, groups_csv=csv, seeds=list(range(N_SEMENTES)))
    cfg2.chronos_t5 = []
    cfg2.chronos_bolt = ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"]
    st2 = HB.Study(cfg2, log=mon)
    mx = K.max_cross_corr(Cm, st2.folds)
    kpis([("grupos", st2.groups.nunique()), ("maior grupo", int(st2.groups.value_counts().iloc[0])),
          ("bacias por fold", "/".join(map(str, st2.meta.fold.value_counts().sort_index().tolist()))),
          ("maior r teste–pool", f"{mx.max():.3f}"), ("mediana", f"{mx.median():.3f}")], "Agrupamento rigoroso")
    st2.run_e1(models=["Persistence", "SeasonalNaive", "Ridge-pool", "LightGBM-pool", "DLinear", "NHITS", "LSTM", "GRU",
                       "Transformer", "PatchTST-S", "PatchTST-M", "PatchTST-L", "PatchTST-XL"])
    st2.run_e3_zeroshot()
    kpis([("execuções E10 salvas", contar(st2, "E1_"))], "E10 concluído")""", "6 · E10 (GPU)"),
    code("""
mon = Monitor("Métricas e tabelas")
st.log = mon
df = A.compute_metrics(st)
out = os.path.join(PASTA_RESULTADOS, "tables")
T9 = K.e9_tables(st, df)
for k, v in T9.items():
    v.to_csv(os.path.join(out, k + ".csv"), index=False)
    tabela(v, k, n=len(v))
if RODAR_E10:
    df2 = A.compute_metrics(st2)
    T10 = K.e10_tables(st2, df2, df_main=df)
    for k, v in T10.items():
        v.to_csv(os.path.join(out, k + ".csv"), index=False)
        tabela(v, k, n=len(v))
painel("Pronto", f"<p class='s'>Tabelas E9_*.csv e E10_*.csv em {out}. Envie a pasta tables (ou só esses CSVs) para o texto.</p>")""",
         "7 · Tabelas"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
path = os.path.join(ROOT, "notebooks", "controles_E9_E10.ipynb")
json.dump(nb, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(path, os.path.getsize(path) // 1024, "KB")
