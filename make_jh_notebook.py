"""Gera notebooks/analises_JHydrol_E11_E12.ipynb: análises pedidas nos pareceres do Journal of Hydrology."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "make_notebook.py"), encoding="utf8").read()
ns = {"__file__": os.path.join(ROOT, "make_notebook.py")}
exec(src.split("cells = [")[0], ns)
PKG, code, md, UI = ns["PKG"], ns["code"], ns["md"], ns["UI"]

SAVE = '''
def salvar(T):
    for k, v in T.items():
        if k.startswith("_"):
            continue
        v.to_csv(os.path.join(TAB, k + ".csv"), index=False)
        tabela(v, k, n=len(v))
'''

cells = [
    md("""# Análises para o Journal of Hydrology (E11 e E12)
Usa os resultados já salvos em `resultados_v2` (benchmark, E9) e acrescenta:

* **E11a** cheias (> q90) e estiagens (< q10) nos horizontes 1, 3, 6 e 12 meses, contra o PAR(p) (só leitura, minutos);
* **E11b** ganho com contexto longo por bacia contra a memória hidrológica da bacia (só leitura, minutos);
* **E12** teste sem contaminação possível: vazões naturais do ONS de 2025–2026, depois do lançamento do Chronos-Bolt (26/11/2024) e do Chronos-2 (20/10/2025) (download do ONS e inferência, ~15 min);
* **E11d** ablação do gerador sintético: sem sazonalidade, sem persistência, sem assimetria (PatchTST-S, GPU, ~1–2 h);
* **E11c** escolha do contexto pelos grupos de validação, sem olhar o teste (refaz o E9 guardando a validação, GPU, ~3–4 h).

Cada bloco grava suas tabelas em `resultados_v2/tables` assim que termina. Se a sessão cair, rode de novo: ele continua de onde parou."""),
    code("""
!pip -q install "chronos-forecasting>=2.2" lightgbm peft pyarrow openpyxl requests
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA (ative a GPU)")""", "1 · Instalar dependências"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
ARQUIVO_VAZOES = "/content/drive/MyDrive/reservatorios/Vazões_Diárias_1931_2024.xlsx"  #@param {type:"string"}
PASTA_RESULTADOS = "/content/drive/MyDrive/reservatorios/resultados_v2"  #@param {type:"string"}
PASTA_ONS = "/content/drive/MyDrive/reservatorios/ons_dados_hidrologicos"  #@param {type:"string"}""",
         "2 · Arquivos (a mesma pasta de resultados do benchmark)"),
    code(f"""
import base64, io, zipfile, sys, os
_PKG = "{PKG}"
os.makedirs("/content/hb", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(_PKG))).extractall("/content/hb")
sys.path.insert(0, "/content/hb")
for m in [k for k in sys.modules if k.startswith("hydrobench")]:
    del sys.modules[m]
import hydrobench as HB
from hydrobench import analysis as A, controls as K, controls2 as K2
{UI}
{SAVE}
painel("Pacote hydrobench carregado", "<p class='s'>inclui hydrobench.controls2 (E11, E12)</p>")""", "3 · Pacote do estudo (oculto)"),
    code("""
N_SEMENTES = 5  #@param {type:"integer"}
RODAR_E12_POS_LANCAMENTO = True  #@param {type:"boolean"}
RODAR_E11D_ABLACAO_SINTETICA = True  #@param {type:"boolean"}
RODAR_E11C_SELECAO_POR_VALIDACAO = True  #@param {type:"boolean"}
cfg = HB.Config(data_path=ARQUIVO_VAZOES, out_dir=PASTA_RESULTADOS, seeds=list(range(N_SEMENTES)))
mon = Monitor("Preparando dados e métricas já salvas")
st = HB.Study(cfg, log=mon)
TAB = os.path.join(PASTA_RESULTADOS, "tables")
df = A.compute_metrics(st)
kpis([("bacias", len(st.basins)), ("grupos", st.groups.nunique()), ("linhas de métricas", f"{len(df):,}")], "Estudo principal")""",
         "4 · Preparar"),
    code("""
mon = Monitor("E11a · cheias e estiagens por horizonte")
st.log = mon
T = K2.extremes(st)
T["_per_basin"].to_csv(os.path.join(TAB, "E11a_extremes_per_basin.csv.gz"), index=False, compression="gzip")
salvar(T)""", "5 · E11a Cheias e estiagens (minutos)"),
    code("""
mon = Monitor("E11b · ganho com contexto longo x memória da bacia")
st.log = mon
salvar(K2.memory_vs_context(st, df))""", "6 · E11b Memória hidrológica (minutos)"),
    code("""
if RODAR_E12_POS_LANCAMENTO:
    mon = Monitor("E12 · teste pós-lançamento com dados do ONS 2025-2026")
    st.log = mon
    r = K2.run_post_release(st, PASTA_ONS)
    if r is not None:
        fc, mt = r
        salvar(K2.post_release_tables(st, fc, mt))
        tabela(mt, "E12_matching (bacia x reservatório do ONS)", n=len(mt))""", "7 · E12 Teste sem contaminação (~15 min)"),
    code("""
if RODAR_E11D_ABLACAO_SINTETICA:
    mon = Monitor("E11d · ablação do gerador sintético (PatchTST-S)")
    st.log = mon
    K2.run_synthetic_ablation(st)
    df = A.compute_metrics(st)
    salvar(K2.synthetic_ablation_table(st, df))""", "8 · E11d Ablação sintética (GPU, ~1–2 h)"),
    code("""
if RODAR_E11C_SELECAO_POR_VALIDACAO:
    mon = Monitor("E11c · contexto escolhido pela validação")
    st.log = mon
    K2.run_e9_validation(st)
    df = A.compute_metrics(st)
    salvar(K2.validation_selection(st, df))""", "9 · E11c Seleção do contexto pela validação (GPU, ~3–4 h)"),
    code("""
mon = Monitor("Exportação por bacia")
st.log = mon
df = A.compute_metrics(st)
K2.export_per_basin(st, df, os.path.join(TAB, "per_basin_metrics_h1_3_6_12.csv.gz"))
painel("Pronto", f"<p class='s'>Tabelas E11*.csv e E12*.csv e o arquivo por bacia estão em {TAB}.</p>")""",
         "10 · Exportar resultados por bacia"),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
path = os.path.join(ROOT, "notebooks", "analises_JHydrol_E11_E12.ipynb")
json.dump(nb, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(path, os.path.getsize(path) // 1024, "KB")
