# Visão Computacional em Faixas de Pedestres para Avaliação de Risco de Acidentes

<p align="center">
  <img src="https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange" alt="Status: Em desenvolvimento">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Visão%20Computacional-green?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLO-Deep%20Learning-purple" alt="YOLO">
</p>

<p align="center">
  Sistema de análise de conflitos entre pedestres e veículos utilizando Visão Computacional e Deep Learning para avaliação da segurança viária.
</p>

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Metodologia](#metodologia)
- [Indicadores de Segurança](#indicadores-de-segurança)
- [Índice de Conflito Observacional](#índice-de-conflito-observacional-ico)
- [Impactos Esperados](#impactos-esperados)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Equipe Desenvolvedora](#equipe-desenvolvedora)
- [Informações Acadêmicas](#informações-acadêmicas)

## Sobre o Projeto

A segurança dos cidadãos nas vias públicas representa um desafio para a gestão da mobilidade urbana no Brasil. Nesse contexto, este projeto propõe o desenvolvimento de um sistema computacional baseado em **Visão Computacional e Deep Learning** para analisar interações entre pedestres e veículos em faixas de travessia urbanas.

Desenvolvido para a disciplina de Fundamentos de Inteligência Artificial, o projeto tem como objetivo identificar e quantificar conflitos viários em travessias urbanas de Aracaju, Sergipe.

A proposta consiste em detectar, rastrear e analisar interações entre pedestres e veículos para identificar situações potencialmente perigosas. Diferentemente das abordagens retrospectivas, que dependem principalmente de registros de acidentes já ocorridos, o sistema busca adotar uma perspectiva preventiva, utilizando situações de quase-acidente e indicadores de conflito para avaliar as condições de segurança das travessias.

## Metodologia

O sistema processa vídeos do tráfego urbano por meio de um pipeline computacional composto por detecção de objetos, rastreamento de trajetórias e análise espaço-temporal das interações entre pedestres e veículos.

A partir dos dados extraídos, são calculadas Medidas Substitutas de Segurança (*Surrogate Safety Measures — SSMs*), utilizadas para caracterizar conflitos viários e subsidiar a avaliação de risco.

O processamento está organizado nas seguintes etapas:

1. **Aquisição de dados:** obtenção de vídeos de travessias urbanas.
2. **Detecção de objetos:** identificação de pedestres e veículos por meio de modelos de Deep Learning.
3. **Rastreamento:** acompanhamento das trajetórias dos objetos detectados ao longo dos quadros.
4. **Análise das interações:** avaliação das relações espaciais e temporais entre os usuários da via.
5. **Cálculo dos indicadores:** extração das métricas utilizadas na avaliação de risco.
6. **Visualização dos resultados:** apresentação dos dados em um painel interativo de apoio à decisão.

## Indicadores de Segurança

O sistema considera três indicadores principais para caracterizar as interações entre pedestres e veículos.

| Indicador | Descrição |
|---|---|
| **Pcrit — Taxa de Tempo Pós-Invasão Crítico** | Baseado no *Post-Encroachment Time* (PET), avalia o intervalo temporal entre a saída de um usuário de uma zona de conflito e a entrada de outro na mesma região. |
| **Enorm — Tempo Médio de Espera Normalizado** | Quantifica a dificuldade de inserção do pedestre na via diante do fluxo contínuo de veículos. |
| **Dfaixa — Taxa de Não-Cedência de Preferência** | Avalia a frequência com que os condutores deixam de ceder passagem diante da intenção de travessia do pedestre. |

Essas métricas são utilizadas como elementos complementares na caracterização das condições de segurança observadas nas travessias.

## Índice de Conflito Observacional (ICO)

O **Índice de Conflito Observacional (ICO)** é um indicador sintético proposto para consolidar as métricas calculadas pelo sistema em uma escala de 0 a 100 pontos.

O índice tem como finalidade facilitar a interpretação dos resultados e classificar as travessias de acordo com o nível de risco observacional identificado.

| Classificação | Interpretação |
|---|---|
| **Baixo** | Menor nível de risco observado. |
| **Moderado** | Presença de situações que exigem atenção. |
| **Crítico** | Maior intensidade ou concentração de conflitos observados. |

O ICO pretende funcionar como um Sistema de Apoio à Decisão (SAD), auxiliando órgãos de trânsito na identificação de locais prioritários para vistorias e intervenções de engenharia e sinalização.

*Nota: a fórmula de composição do índice e os limites numéricos de cada categoria devem ser definidos e validados metodologicamente. O ICO representa uma estimativa de risco observacional, não uma probabilidade direta de ocorrência de acidentes.*

## Impactos Esperados

### Segurança Viária

Identificação antecipada de situações potencialmente perigosas, como proximidade crítica entre usuários, conflitos de trajetória e variações bruscas de velocidade, reduzindo a dependência exclusiva de registros de acidentes consumados.

### Gestão e Planejamento Urbano

Produção de dados estruturados e quantitativos para apoiar a Superintendência Municipal de Transportes e Trânsito (SMTT) e outros gestores públicos na priorização de melhorias na infraestrutura viária, na sinalização e nas ações de fiscalização.

### Desenvolvimento Acadêmico e Científico

Aplicação prática de algoritmos de Inteligência Artificial a problemas concretos de mobilidade urbana, contribuindo para a avaliação de métodos computacionais e para o desenvolvimento de pesquisas futuras na região.

## Tecnologias Utilizadas

O projeto utiliza ferramentas de código aberto para as etapas de processamento visual, análise de dados e visualização dos resultados.

| Categoria | Tecnologias | Aplicação |
|---|---|---|
| Visão Computacional | OpenCV | Processamento de imagens e vídeos. |
| Detecção de Objetos | Ultralytics YOLO | Identificação de pedestres e veículos. |
| Rastreamento | ByteTrack | Associação e acompanhamento de objetos entre quadros. |
| Anotação de Dados | CVAT ou Label Studio | Rotulagem de imagens e construção do conjunto de validação (*ground truth*). |
| Computação Científica | NumPy e SciPy | Operações numéricas e cálculos científicos. |
| Análise de Dados | Pandas | Organização e análise dos dados extraídos. |
| Análise Geométrica | Shapely | Representação e análise espacial de trajetórias e zonas de conflito. |
| Visualização | Matplotlib | Geração de gráficos e visualizações analíticas. |
| Dashboard | Streamlit | Desenvolvimento do painel interativo de apoio à decisão. |

## Equipe Desenvolvedora

Projeto desenvolvido sob a orientação do **Prof. Dr. Victor Flavio**.

- André Conceição da Silva Filho
- João Paulo Amorim Bastos
- Michael Alves dos Santos
- Rafael José Coelho Souza

## Informações Acadêmicas

**Universidade Federal de Sergipe (UFS)**  
Centro de Ciências Exatas e Tecnologia (CCET)  
Departamento de Computação (DCOMP)

> **Disciplina:** Fundamentos de Inteligência Artificial  
**Período:** Setembro de 2026

## Calibra??o dos detectores

Execute com Python 3.9+ e as depend?ncias `ultralytics` e `opencv-python` instaladas:

```powershell
python src/perception/findPedestrian.py --source data/raw/TwoWayTraffic720p.mp4
python src/perception/findVehicle.py --source data/raw/BackVehcilesTraffic720p.mp4
python src/perception/findPedestrian.py --source data/raw/TwoWayTraffic720p.mp4 --imgsz 960 --conf 0.10
```

Os modelos originais foram mantidos: `yolo26n.pt` para pessoas e `yolo11n.pt` para ve?culos. Os pesos podem ser baixados pelo Ultralytics na primeira execu??o. Use `--model` para selecionar outros pesos compat?veis com as classes COCO. Ve?culos incluem carros, motos, ?nibus e caminh?es; bicicletas n?o est?o inclu?das.

O corte de confian?a de 0,10 permite encaminhar detec??es fracas ao BoT-SORT para recuperar trajet?rias; os limiares internos do rastreador continuam controlando a associa??o e a cria??o de IDs. Esse valor n?o garante reconhecimento. A resolu??o padr?o permanece 640; compare com 960 quando houver objetos pequenos ou distantes, observando o custo de processamento. `--iou` controla a supress?o de caixas do detector, n?o a associa??o do rastreador.

Use `--no-show` para executar sem janela, `--save` para salvar o v?deo anotado e `--no-save` para desativar a grava??o. A grava??o est? habilitada por padr?o apenas no detector de ve?culos. A tecla `q` encerra a visualiza??o.

Antes de adotar uma calibra??o, compare os mesmos trechos anotados, incluindo objetos distantes, oclus?es e ilumina??o vari?vel. Registre falsos positivos, objetos perdidos e trocas de ID; compare precis?o e recall por classe. Aumentar a confian?a pode reduzir falsos positivos, mas tamb?m remove detec??es ?teis ? recupera??o de trajet?rias. Os valores fornecidos s?o um ponto de partida, sem valida??o de acur?cia no conjunto local.

Refer?ncia dos par?metros: [documenta??o oficial de rastreamento Ultralytics](https://docs.ultralytics.com/modes/track/).
