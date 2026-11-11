# PedestrianConflict-SE

## Guia técnico de implementação, validação e organização do trabalho

**Versão:** 0.1 --- proposta de documentação técnica\
**Finalidade:** orientar a equipe desde a preparação do ambiente até uma
primeira execução ponta a ponta, com critérios de aceitação e sugestões
de issues para o Kanban existente.

> **Nota de integração:** este documento propõe conteúdo para o
> repositório já existente. Antes de criar arquivos ou issues, confira a
> estrutura, os nomes das pastas, as convenções de branch, os rótulos e
> os milestones atuais. Não crie um segundo quadro Kanban nem duplique
> issues que já existam. Adapte os identificadores sugeridos abaixo aos
> padrões do projeto.

------------------------------------------------------------------------

## 1. Objetivo, estágio atual e escopo temporal

O PedestrianConflict-SE investiga uma abordagem de análise observacional de interações entre pedestres e outros usuários da via, visando apoiar a priorização técnica de vistorias. O projeto transforma observações visuais em trajetórias, trajetórias em eventos mensuráveis e eventos em indicadores que possam ser auditados.

### 1.1 Estado real no Check-in 2

O Check-in 2 não deve ser apresentado como uma demonstração de software de visão computacional em execução. A prova de conceito atual combina:

1. **Vídeo proxy editado no CapCut:** representação visual dos agentes com caixas delimitadoras e demarcação gráfica das regiões de interesse (ROIs).
2. **Planilha sintética de trajetórias:** dados tabulares com tempos, posições e velocidades de referência para validar as fórmulas e o fluxo numérico antes da automação.
3. **Regras determinísticas de decisão:** aplicação planejada de limiares e regras para demonstrar a lógica analítica, sem alegar que o resultado foi inferido automaticamente do vídeo.
4. **Fundamentação metodológica:** descrição da futura transformação de percepção visual em eventos, indicadores e sumário observacional.

As caixas e zonas desenhadas no vídeo proxy são anotações gráficas demonstrativas, não predições produzidas por um detector. A planilha sintética testa a lógica numérica, não comprova a precisão de um detector, rastreador ou homografia aplicada a imagens reais.

### 1.2 Escopo temporal: Check-in 2 versus Check-in 3

**Check-in 2 — sustentação conceitual e prova de conceito analítica**
- Explicar o problema, a fundamentação e as métricas propostas.
- Demonstrar o conceito visual com o vídeo proxy e as ROIs desenhadas.
- Demonstrar o fluxo numérico com dados sintéticos e regras determinísticas.
- Explicitar a contingência para a pendência de autorização dos vídeos oficiais.
- Identificar limitações e separar claramente resultados demonstrativos de resultados automatizados.

**Check-in 3 — implementação e evidência quantitativa inicial**
- Formalizar contratos de dados e protocolo experimental.
- Construir uma amostra piloto anotada como ground truth.
- Executar um detector baseline pré-treinado da família YOLO.
- Implementar Centroid Tracking como baseline de rastreamento.
- Desenvolver o avaliador de detecção e reportar IoU, Precisão, Revocação e F1-Score.
- Validar homografia com gabaritos métricos e testar o cálculo de PET com dados tabulares.
- Integrar os módulos que estiverem prontos e documentar resultados e limitações.

### 1.3 Fora do escopo inicial

Não afirmar, antes de haver evidência:
- que o pipeline de visão computacional está executando;
- que as caixas do vídeo proxy foram geradas automaticamente;
- que dados sintéticos validam a percepção visual em vídeo real;
- que o sistema estima probabilidade absoluta de colisão;
- que o sistema já generaliza para diferentes câmeras e locais;
- que há desempenho em tempo real antes de medi-lo.

O **Índice de Conflito Observacional (ICO)** é um indicador observacional de suporte à priorização técnica de vistorias. Não representa probabilidade absoluta de acidentes, previsão individual de colisão ou medida universal de risco.

### 1.4 Baseline planejado para o Check-in 3

-   **Detecção:** detector de estágio único, com versão, pesos e
    configuração registrados.
-   **Rastreamento:** Centroid Tracking como referência simples.
-   **Eventos:** regras geométricas e temporais determinísticas.
-   **Risco operacional:** PET crítico com limiar inicial documentado,
    por exemplo `PET < 2.0 s`, sem tratá-lo como fronteira universal.
-   **Avaliação:** métricas de detecção, inspeção visual, testes
    sintéticos de geometria e testes de consistência temporal.

ByteTrack, DeepSORT e outras arquiteturas permanecem candidatas a
comparação. Não se deve mudar o baseline sem registrar a configuração
anterior e o motivo da alteração.

------------------------------------------------------------------------

## 2. Requisitos funcionais

  -----------------------------------------------------------------------
  ID                      Requisito               Evidência mínima de
                                                  aceitação
  ----------------------- ----------------------- -----------------------
  RF-01                   Detectar classes de     Predições exportadas
                          agentes definidas no    por quadro com caixa,
                          protocolo               classe e confiança

  RF-02                   Rastrear agentes ao     Trajetórias com
                          longo do tempo          `track_id`, timestamps
                                                  e tratamento de perdas

  RF-03                   Definir ROIs e          Eventos geométricos
                          registrar               reproduzíveis em vídeo
                          entradas/saídas         e testes sintéticos

  RF-04                   Converter coordenadas   Calibração documentada
                          de imagem para plano    e erro de reprojeção
                          viário quando calibrado medido

  RF-05                   Calcular PET para pares Casos de teste com
                          de ocupações elegíveis  resultado esperado e
                                                  eventos inválidos
                                                  explicitamente marcados

  RF-06                   Calcular espera e       Definições
                          não-cedência conforme   operacionais,
                          protocolo               denominadores e
                                                  política para dados
                                                  ausentes documentados

  RF-07                   Agregar indicadores por Resumo rastreável até
                          sessão                  os eventos individuais

  RF-08                   Avaliar detecção e,     Relatório quantitativo
                          quando houver           reproduzível
                          anotações,              
                          rastreamento/eventos    

  RF-09                   Reproduzir uma execução Arquivo de
                          por configuração        configuração, versões e
                          registrada              instruções de execução
  -----------------------------------------------------------------------

### 2.1 Requisitos não funcionais

-   **Reprodutibilidade:** registrar versão do código, pesos do modelo,
    dependências, parâmetros, vídeo e calibração.
-   **Robustez:** tratar quadros sem detecção, caixas inválidas,
    timestamps duplicados, trilhas fragmentadas e eventos incompletos.
-   **Auditabilidade:** conservar registros individuais de eventos, além
    dos agregados.
-   **Privacidade:** limitar acesso a vídeos e evitar dados pessoais
    desnecessários nos logs.
-   **Desempenho:** medir latência, FPS, memória, resolução e hardware;
    não prometer tempo real sem evidência.
-   **Modularidade:** permitir substituir detector ou rastreador sem
    reescrever cinemática e decisão.

------------------------------------------------------------------------

## 3. Arquitetura proposta

``` text
Vídeo + metadados
       |
       v
src/perception/  -> detecções por quadro
       |
       v
src/tracking/    -> trajetórias com track_id
       |
       v
src/kinematics/  -> posições, velocidades e eventos geométricos
       |
       v
src/decision/    -> PET, espera, não-cedência, ICO
       |
       v
Avaliação + relatórios + visualizações
```

Estrutura de diretórios sugerida --- adaptar à organização já existente:

``` text
docs/
  guia_tecnico.md
  protocolo_anotacao.md
  protocolo_experimental.md
  arquitetura.md
contracts/
  detection.schema.json
  trajectory.schema.json
  event.schema.json
configs/
  cameras/
  experiments/
data/
  raw/              # normalmente fora do Git
  annotations/
  synthetic/
  processed/        # normalmente fora do Git
src/
  perception/
  tracking/
  kinematics/
  decision/
tests/
  fixtures/
  unit/
  integration/
reports/
  evaluation/
```

Não é necessário criar todas as pastas imediatamente. Crie somente as
que tiverem responsável, entrega definida e uso concreto.

------------------------------------------------------------------------

## 4. Preparação do ambiente para o Check-in 3

Esta seção é um plano de preparação, não uma indicação de que a inferência já foi executada. Manter o ambiente inicial enxuto; adiar otimizações de GPU e configurações avançadas até que o pipeline básico funcione localmente ou em ambiente de nuvem autorizado.

### 4.1 Pré-requisitos

-   Git e acesso ao repositório existente.
-   Python 3.10 ou 3.11, conforme compatibilidade das dependências
    escolhidas.
-   Ambiente virtual Python.
-   Editor (por exemplo, VS Code).
-   Espaço em disco para vídeos e resultados.
-   Opcional: GPU NVIDIA compatível e drivers/CUDA adequados para a
    versão do PyTorch escolhida.

**Importante:** a instalação de PyTorch com suporte a GPU varia conforme
sistema operacional, driver e versão CUDA. Use o seletor oficial do
PyTorch para gerar o comando apropriado. Não copie um comando CUDA
aleatório de uma máquina diferente.

### 4.2 Clonar ou atualizar o repositório

Se ainda não houver uma cópia local:

``` bash
git clone <URL_DO_REPOSITORIO>
cd <DIRETORIO_DO_REPOSITORIO>
git status
```

Se o repositório já estiver clonado:

``` bash
cd <DIRETORIO_DO_REPOSITORIO>
git status
git pull
```

Antes de atualizar, verifique alterações locais. Não descarte trabalho
com `git reset --hard` ou comandos destrutivos sem confirmar que ele
está salvo.

### 4.3 Criar ambiente virtual

**Windows --- PowerShell:**

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux/WSL/macOS:**

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Se a ativação do PowerShell for bloqueada por política corporativa, siga
a orientação da administração da máquina; não altere políticas globais
de segurança sem necessidade.

### 4.4 Instalar dependências iniciais

Uma base mínima para contratos, planilhas, geometria e avaliação:

``` bash
python -m pip install numpy pandas scipy matplotlib opencv-python pyyaml jsonschema pytest jupyter
```

Quando a equipe iniciar a implementação do detector YOLO, adicionar a dependência de inferência:

``` bash
python -m pip install ultralytics
```

Instale o PyTorch conforme as instruções oficiais adequadas ao ambiente
antes de diagnosticar problemas de aceleração. A instalação do pacote
`ultralytics` não garante, por si só, que CUDA esteja configurado.

Se o projeto escolher outro detector, mantenha a dependência
correspondente no arquivo de dependências do repositório em vez de
instalar várias famílias sem uso.

### 4.5 Registrar dependências

Depois de confirmar que o ambiente funciona:

``` bash
python --version
python -m pip freeze > requirements-lock.txt
```

A equipe pode manter `requirements.txt` com dependências diretas e um
arquivo de lock separado, conforme a convenção do projeto. Evite
sobrescrever arquivos de dependências existentes sem revisar as decisões
já adotadas.

Para reproduzir o ambiente registrado:

``` bash
python -m pip install -r requirements-lock.txt
```

Esse arquivo pode variar por sistema operacional e GPU. Documente as
limitações se o lock tiver sido gerado em uma plataforma específica.

### 4.6 Testar as instalações

``` bash
python -c "import cv2, numpy, pandas, scipy, matplotlib, yaml; print('Dependências básicas OK')"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
pytest -q
```

O último comando pode não encontrar testes na primeira configuração.
Nesse caso, registre que ainda não há testes, em vez de interpretar
ausência de testes como sucesso do pipeline.

### 4.7 Ferramentas de anotação

Escolher **uma ferramenta principal** para o primeiro ciclo, conforme os
recursos e as políticas de dados:

-   **CVAT:** útil para caixas, vídeo e tarefas de anotação com suporte
    a rastreamento, dependendo da configuração.
-   **Label Studio:** alternativa de anotação com diferentes formatos e
    tarefas.

Começar por uma instância hospedada pode reduzir o custo inicial de
instalação. Para dados restritos, avaliar instalação local e controles
de acesso antes de enviar qualquer vídeo a um serviço externo. O formato
de exportação precisa ser testado com um pequeno exemplo antes de anotar
em escala.

------------------------------------------------------------------------

## 5. Estratégia de dados e vídeo proxy

### 5.1 Evidência visual disponível para o Check-in 2

O vídeo editado no CapCut serve como **prova de conceito visual**: comunica onde a equipe pretende detectar agentes e quais ROIs serão analisadas. Caixas delimitadoras, rótulos e polígonos adicionados na edição devem ser descritos como elementos gráficos inseridos manualmente, não como saídas de inferência.

A planilha sintética de trajetórias serve para verificar a lógica temporal e numérica com valores controlados. Ela deve conter timestamps, posições e, quando aplicável, velocidades esperadas. Deve ser possível recalcular manualmente pelo menos alguns casos e confirmar que o resultado do script ou da planilha corresponde ao esperado.

Para o Check-in 2, apresentar em conjunto:
- vídeo proxy com ROIs identificadas;
- planilha sintética e fórmulas;
- exemplo de regra determinística e resultado esperado;
- limitações e distinção explícita entre demonstração manual, dado sintético e futura automação.

### 5.2 Critérios para escolher dados do Check-in 3


### 5.2 Critérios para escolher o material

Para cada vídeo, registrar: - `video_id` único; - origem e URL de
referência; - autor ou organização publicadora, quando conhecido; -
licença e condições de uso; - data de obtenção; - duração, FPS e
resolução; - câmera fixa ou móvel; - localização aproximada e descrição
da cena, quando disponíveis; - restrições de privacidade; - motivo de
inclusão no conjunto.

Priorizar material que permita observar a área de travessia, a calçada e
as interações relevantes. Um vídeo sem visibilidade da região de
conflito pode ser útil para detecção, mas inadequado para validar PET.

### 5.3 Fontes possíveis

1.  **Vídeos institucionais autorizados:** potencialmente mais alinhados
    ao cenário local, mas dependem da autorização formal, das condições
    de uso e da análise de privacidade.
2.  **Conjuntos de dados de pesquisa:** procurar conjuntos com vídeos
    urbanos e termos de licença documentados, como MOTChallenge, JAAD ou
    BDD100K, verificando individualmente se o conteúdo atende ao caso de
    uso e às condições de acesso.
3.  **Vídeos abertos com licença explícita:** confirmar a licença do
    vídeo específico; a disponibilidade pública não significa permissão
    irrestrita de download, redistribuição ou uso.
4.  **Vídeos sintéticos ou simulados:** úteis para testar a lógica, mas
    não substituem a avaliação de percepção em vídeo real.

Não utilizar gravações institucionais, rostos identificáveis ou dados
restritos em serviços externos sem autorização apropriada. Não publicar
vídeos brutos no Git.

### 5.4 Procedimento para obter um vídeo autorizado

1.  Abra a página oficial do conjunto ou do detentor do vídeo.
2.  Leia a licença, os termos de uso e as regras de citação.
3.  Baixe o arquivo por um link oficial ou método expressamente
    permitido.
4.  Salve o arquivo fora do repositório, por exemplo em `data/raw/`.
5.  Registre a procedência em um manifesto, sem embutir dados pessoais.
6.  Calcule um hash para identificar o arquivo e detectar substituições.
7.  Verifique se o vídeo abre e registre resolução, FPS e duração.

Exemplo para calcular hash SHA-256 com Python:

``` bash
python -c "import hashlib,sys; p=sys.argv[1]; h=hashlib.sha256(); f=open(p,'rb'); [h.update(b) for b in iter(lambda:f.read(1024*1024),b'')]; print(h.hexdigest())" "caminho/do/video.mp4"
```

Exemplo de manifesto CSV:

``` csv
video_id,source,license,local_path,sha256,notes
test_001,URL_OFICIAL,LICENSE_TO_VERIFY,data/raw/test_001.mp4,HASH_A_PREENCHER,trecho de teste
```

Use o endereço real e a licença efetivamente verificada; não deixe
valores de exemplo em um manifesto tratado como definitivo.

### 5.5 Verificar o vídeo com OpenCV

``` python
import cv2

video_path = "data/raw/test_001.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps and fps > 0 else None

print({
    "fps_reportado": fps,
    "quadros": frame_count,
    "largura": width,
    "altura": height,
    "duracao_s": duration,
})

ok, frame = cap.read()
if not ok:
    raise RuntimeError("O arquivo abriu, mas o primeiro quadro não pôde ser lido.")

cv2.imwrite("first_frame_check.jpg", frame)
cap.release()
```

Verifique visualmente se a imagem exportada está correta. FPS reportado
por um arquivo pode não refletir timestamps variáveis; quando o tempo
exato for importante, preserve timestamps da fonte ou documente a
aproximação utilizada.

### 5.6 Cuidados com download

Não assumir que um vídeo pode ser baixado só porque está disponível em
uma plataforma. Use o mecanismo oficial de download, uma API autorizada
ou um link direto fornecido pelo detentor. Ferramentas como `yt-dlp` só
devem ser usadas quando o download for permitido pelos termos da
plataforma e pela licença do conteúdo. Não contorne controles de acesso
nem restrições técnicas.

------------------------------------------------------------------------

## 6. Anotação e ground truth

### 6.1 Começar com um piloto

Antes de anotar todo o conjunto: 1. Selecionar poucos trechos
representativos. 2. Definir classes e regras de anotação. 3. Dois
integrantes anotam parte do mesmo material, de forma independente. 4.
Comparar divergências e revisar o protocolo. 5. Só então ampliar a
anotação.

### 6.2 Protocolo mínimo de detecção

Restringir a taxonomia às classes realmente necessárias para o escopo do projeto. Documentar o mapeamento entre nomes internos, IDs de classe e o padrão COCO quando aplicável; não assumir que uma exportação YOLO usa automaticamente a taxonomia COCO.

Documentar: - classes incluídas e excluídas; - caixa delimitadora e
convenção de coordenadas; - critérios para objetos parcialmente
ocultos; - objetos truncados na borda da imagem; - casos ambíguos; -
tratamento de quadros sem agentes; - versão do protocolo e identidade
operacional do anotador.

### 6.3 Ground truth temporal

O formato YOLO convencional é adequado para caixas por imagem, mas não
basta sozinho para representar identidades persistentes, eventos e PET.
Preservar, quando necessário: - `video_id`, `frame_id` e
`timestamp_s`; - identidade de referência (`gt_track_id`); - classe e
caixa; - região de conflito; - instantes de entrada e saída; -
identificador do evento; - anotação de incerteza ou ambiguidade.

Separar o conjunto de avaliação do material utilizado para ajustar
parâmetros. Evitar distribuir quadros vizinhos do mesmo vídeo entre
treino e teste, pois isso pode inflar artificialmente os resultados.

------------------------------------------------------------------------

## 7. Contratos de dados

Os contratos devem ser independentes do detector e do rastreador
escolhidos. A equipe deve aprovar os campos, os tipos, as unidades e as
regras de valores ausentes antes de implementar a integração.

### 7.1 Contrato A --- detecção por quadro

``` python
{
    "video_id": "test_001",
    "frame_id": 120,
    "timestamp_s": 4.0,
    "detections": [
        {
            "bbox_xyxy": [120.5, 80.0, 165.0, 210.0],
            "class_name": "person",
            "class_id": 0,
            "confidence": 0.93
        }
    ]
}
```

Regras: - `frame_id` é monotônico dentro do vídeo; - `timestamp_s`
representa o instante da observação, não o tempo de execução do
programa; - caixa usa `[x1, y1, x2, y2]`; - caixas inválidas devem ser
rejeitadas ou sinalizadas; - `detections: []` é válido; - documentar se
as coordenadas são pixels absolutos ou normalizadas.

### 7.2 Contrato B --- trajetórias para cinemática

Colunas tabulares mínimas recomendadas:

  -----------------------------------------------------------------------
  Campo                               Significado
  ----------------------------------- -----------------------------------
  `video_id`                          Sessão de origem

  `frame_id`                          Quadro de origem

  `timestamp_s`                       Instante observado

  `track_id`                          Identificador operacional do
                                      rastreador

  `class_name`                        Classe do agente

  `u`, `v`                            Ponto representativo na imagem

  `X_m`, `Y_m`                        Coordenadas métricas, nulas se não
                                      calibradas

  `calibration_valid`                 Validade da conversão para o plano

  `detection_confidence`              Confiança da detecção de origem, se
                                      disponível
  -----------------------------------------------------------------------

Um `track_id` não comprova a identidade física de um agente. Ele
identifica uma trajetória estimada pelo algoritmo dentro do escopo
definido pelo contrato.

### 7.3 Contrato C --- eventos e agregados

Preservar **dois níveis**: 1. Registro individual de evento, para
auditoria e recálculo. 2. Resumo agregado por sessão, para relatórios e
ICO.

O evento individual deve conter, quando aplicável: - `event_id`,
`video_id` e agentes envolvidos; - ROI e regra que disparou o evento; -
tempos de entrada e saída; - PET e validade do cálculo; - motivo de
exclusão ou invalidade; - versão do algoritmo e configuração.

O resumo pode conter: - total de travessias; - eventos elegíveis; - PETs
válidos; - contagem e proporção de PET crítico; - espera média e número
de observações válidas; - eventos elegíveis para não-cedência; -
indicadores de qualidade e cobertura dos dados.

------------------------------------------------------------------------

## 8. Geometria, cinemática e eventos

### 8.1 Homografia

A homografia planar relaciona pontos da imagem a pontos do plano viário:

\[ `\lambda [X,Y,1]`{=tex}\^T = H\[u,v,1\]\^T \]

Requisitos: - obter pontos correspondentes com coordenadas físicas
conhecidas; - registrar a matriz de calibração e a câmera associada; -
medir o erro de reprojeção; - definir origem e orientação dos eixos; -
usar um ponto representativo próximo ao contato com o solo, como a base
da caixa; - sinalizar coordenadas métricas inválidas quando não houver
calibração.

A homografia não recupera a posição 3D completa de um pedestre. Uma
coordenada sintética em metros testa a lógica de cinemática, não a
precisão da homografia aplicada a vídeo real.

### 8.2 Velocidade

Para posições métricas válidas e tempos distintos:

\[ v_t =
`\frac{\sqrt{(X_t-X_{t-1})^2+(Y_t-Y_{t-1})^2}}{t_t-t_{t-1}}`{=tex} \]

Tratar timestamps duplicados, intervalos longos, detecções espúrias e
fragmentações. Não calcular velocidade física a partir de pixels sem
calibração.

### 8.3 PET

Para dois agentes que ocupam a mesma região de conflito em sequência:

\[ PET = t\_{`\mathrm{entrada}`{=tex},B} - t\_{`\mathrm{saída}`{=tex},A}
\]

O cálculo exige uma regra de associação entre agentes e uma definição
inequívoca da região. Eventos simultâneos, entradas/saídas não
observadas e trajetórias fragmentadas devem ser sinalizados, não
transformados silenciosamente em PET válido.

O limiar inicial `PET < 2.0 s` é uma escolha operacional para testar o
pipeline, não uma fronteira universal entre seguro e inseguro.

### 8.4 Espera e não-cedência

Uma possível definição para espera normalizada é:

\[ E\_{`\mathrm{norm}`{=tex}} =
`\frac{\overline{T}_{\mathrm{espera}}}{T_{\mathrm{referência}}}`{=tex}
\]

Definir começo/fim da espera e o denominador positivo. Para
não-cedência:

\[ D\_{`\mathrm{faixa}`{=tex}} =
`\frac{N_{\mathrm{eventos\ sem\ cessão}}}{N_{\mathrm{eventos\ elegíveis}}}`{=tex}
\]

Definir cessão, elegibilidade e tratamento de eventos ambíguos. Se não
houver eventos elegíveis, registrar como indefinido, não como zero.

------------------------------------------------------------------------

## 9. Avaliação quantitativa

### 9.1 IoU e métricas de detecção

\[ IoU = `\frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}`{=tex} \]

Estabelecer previamente o limiar de IoU e a regra de correspondência
um-para-um, considerando também a classe. Exemplo inicial: IoU ≥ 0,50,
se adequado ao protocolo.

\[ Precisão = `\frac{TP}{TP+FP}`{=tex} `\qquad`{=tex} Revocação =
`\frac{TP}{TP+FN}`{=tex} \]

\[ F_1 = 2`\frac{Precisão \cdot Revocação}{Precisão+Revocação}`{=tex} \]

Tratar denominadores zero explicitamente. Reportar métricas por classe e
inspecionar falsos positivos e falsos negativos, em vez de depender
apenas do F1 agregado.

### 9.2 Rastreamento e eventos

IoU, precisão, revocação e F1 não medem integralmente a continuidade das
identidades. Se o ground truth permitir, considerar IDF1, HOTA ou MOTA,
com implementação compatível com o formato de anotação.

Para eventos, avaliar: - eventos encontrados e perdidos; - eventos
duplicados; - erro nos instantes de entrada/saída; - erro absoluto do
PET; - proporção de eventos sem dados suficientes; - concordância entre
anotadores.

### 9.3 Baseline reproduzível

Cada execução deve registrar: - commit ou versão do código; - versão do
detector e pesos; - rastreador e parâmetros; - versão do conjunto de
dados; - resolução e FPS; - arquivo de calibração; - limiares de
detecção, associação e PET; - métricas calculadas e limitações
conhecidas.

------------------------------------------------------------------------

## 10. ICO --- índice observacional composto

Uma formulação preliminar pode ser:

\[ ICO = w_1R\_{`\mathrm{PET}`{=tex}} + w_2E\_{`\mathrm{norm}`{=tex}} +
w_3D\_{`\mathrm{faixa}`{=tex}} \]

com pesos iniciais `w1 = 0.50`, `w2 = 0.25`, `w3 = 0.25`, desde que os
indicadores estejam normalizados e orientados de forma coerente.

Antes de publicar o índice: 1. Definir exatamente `R_PET`, por exemplo,
proporção de eventos elegíveis com PET abaixo do limiar. 2. Colocar os
componentes em escalas comparáveis. 3. Definir política para indicadores
ausentes. 4. Manter versão baseline com pesos fixos. 5. Fazer análise de
sensibilidade dos pesos em ±10% e ±20%. 6. Registrar se a ordenação das
sessões muda. 7. Descrever o ICO como indicador observacional
exploratório, a menos que haja validação independente que sustente uma
interpretação mais forte.

AHP pode documentar julgamentos de especialistas e auxiliar a selecionar
pesos, mas não garante que o índice seja objetivamente correto.
Registrar julgadores, comparações, agregação e consistência dos
julgamentos.

------------------------------------------------------------------------

## 11. Testes mínimos antes da integração

### 11.1 Testes sintéticos de geometria e PET

Construir trajetórias simples com tempos conhecidos. Exemplos: - agente
A sai em 2,0 s e B entra em 3,5 s → PET esperado de 1,5 s; - B entra
antes da saída de A → evento simultâneo ou sobreposto, tratado pela
regra do protocolo; - entrada ou saída ausente → PET inválido; - nenhuma
travessia → contagens zero e razões sem denominador marcadas como
indefinidas; - timestamps repetidos → velocidade inválida naquele
intervalo; - coordenadas não calibradas → não produzir velocidade em
m/s.

### 11.2 Testes de detecção

-   caixa perfeita;
-   caixa parcialmente sobreposta;
-   classe incorreta;
-   predição extra;
-   objeto omitido;
-   quadro sem objetos;
-   caixa degenerada ou fora dos limites.

### 11.3 Testes de integração

Executar um vídeo curto e verificar que: - o leitor identifica metadados
e quadros; - as detecções obedecem ao Contrato A; - o rastreador produz
IDs e timestamps; - a cinemática sinaliza quando a calibração não
existe; - os eventos contêm a origem e a validade; - o resumo agregado é
consistente com os eventos; - o relatório registra configuração e
limitações.

------------------------------------------------------------------------

## 12. Responsabilidades operacionais por dupla

A distribuição deve seguir a divisão já consolidada pela equipe. As atividades abaixo refinam o escopo das duplas, sem criar frentes paralelas.

### João Pedro e Rafael — curadoria, ground truth, contratos e avaliação

- Curar uma amostra piloto enxuta de aproximadamente 100–300 quadros e registrar origem, licença, critérios de seleção e limitações.
- Definir protocolo de rotulagem para as classes de interesse, com mapeamento documentado para COCO quando aplicável.
- Anotar o lote piloto no CVAT ou Label Studio e validar a exportação YOLO.
- Governar os contratos de dados e manter fixtures/mocks para testes desacoplados.
- Desenvolver o avaliador de detecção com IoU, Precisão, Revocação e F1-Score.
- Preparar relatório com métricas, erros representativos e limitações do ground truth.

### André e Michael — percepção, rastreamento e cinemática

- Executar o detector baseline pré-treinado da família YOLO quando o ambiente e o conjunto piloto estiverem prontos.
- Implementar Centroid Tracking como referência inicial anterior à comparação com ByteTrack.
- Formalizar a transformação planar por homografia com gabaritos métricos oficiais ou pontos físicos conhecidos.
- Calcular entradas e saídas das ROIs e testar PET crítico com dados tabulares controlados.
- Exportar saídas conforme os contratos A e B e registrar versões e parâmetros.
- Documentar limitações de calibração, oclusão, fragmentação de trajetórias e timestamps.

### Integração compartilhada

As duas duplas revisam em conjunto os contratos, os exemplos de entrada/saída e os critérios de aceitação. Alterações de esquema devem ser comunicadas antes de serem incorporadas por um módulo. A conclusão de uma tarefa exige evidência verificável, não apenas código escrito ou uma visualização ilustrativa.

---

## 13. Issues sugeridas para o Kanban existente

Os títulos abaixo são propostas de issues, não afirmações de que elas já
existem. Primeiro procure por issues equivalentes e reutilize-as quando
apropriado. Use os rótulos, responsáveis, milestones e convenções já
adotados pelo repositório.

### Epic / marco técnico A --- Contratos e protocolo

**Issue A1 --- Formalizar contratos de detecção, trajetória e evento** -
**Objetivo:** fixar campos, tipos, unidades e política de valores
ausentes. - **Tarefas:** documentar os três contratos; criar exemplos
válidos e inválidos; validar um mock de cada módulo. - **Aceitação:**
módulos conseguem consumir fixtures sem depender de detector real;
mudanças de contrato são versionadas. - **Dependências:** nenhuma.

**Issue A2 --- Definir protocolo de anotação e ground truth** -
**Objetivo:** garantir anotações consistentes. - **Tarefas:** classes,
caixas, oclusões, IDs, ROIs, entradas/saídas e ambiguidades; executar
piloto com dois anotadores. - **Aceitação:** protocolo versionado e
divergências do piloto resolvidas/documentadas. - **Dependências:**
seleção preliminar de vídeos.

**Issue A3 --- Definir protocolo experimental e divisão de dados** -
**Objetivo:** evitar comparação injusta e vazamento entre conjuntos. -
**Tarefas:** definir unidade de separação (vídeo/sessão/câmera),
métricas e configurações registradas. - **Aceitação:** documento permite
reproduzir a comparação do baseline.

### Epic / marco técnico B --- Ambiente e dados

**Issue B1 --- Documentar ambiente Python e dependências** -
**Objetivo:** permitir que outro integrante prepare o ambiente. -
**Tarefas:** versão Python, ambiente virtual, dependências, instalação
de PyTorch e teste de imports. - **Aceitação:** outro integrante executa
os comandos documentados e registra o resultado.

**Issue B2 --- Definir política de dados, licenças e armazenamento** -
**Objetivo:** evitar uso ou publicação indevida de vídeos. -
**Tarefas:** critérios de seleção, registro de origem/licença,
localização de armazenamento e `.gitignore`. - **Aceitação:** cada vídeo
usado tem origem e condição de uso registradas; vídeos brutos não são
commitados por engano.

**Issue B3 --- Obter e inspecionar amostra de vídeo autorizada** -
**Objetivo:** estabelecer um caso de teste real. - **Tarefas:** baixar
por fonte permitida; registrar hash e metadados; verificar abertura e
qualidade visual. - **Aceitação:** vídeo abre, metadados estão
documentados e a ROI é observável. - **Dependências:** B2.

**Issue B4 --- Configurar ferramenta de anotação** - **Objetivo:**
produzir ground truth no formato necessário. - **Tarefas:** escolher
CVAT ou Label Studio; testar exportação; validar conversão para o
esquema interno. - **Aceitação:** pequena amostra exportada e validada
automaticamente. - **Dependências:** A1 e A2.

### Epic / marco técnico C --- Percepção e avaliação

**Issue C1 --- Implementar avaliador de detecção com IoU** -
**Objetivo:** comparar predições com ground truth. - **Tarefas:**
correspondência um-para-um; TP/FP/FN; precisão, revocação, F1; testes de
casos extremos. - **Aceitação:** testes sintéticos passam e o relatório
explicita limiar de IoU e política de denominadores. - **Dependências:**
A1 e A2.

**Issue C2 --- Executar e documentar detector baseline** - **Objetivo:**
produzir uma referência mensurada. - **Tarefas:** fixar
versão/pesos/configuração; executar no conjunto piloto; salvar predições
e exemplos de erros. - **Aceitação:** resultados reproduzíveis com
configuração e métricas registradas. - **Dependências:** B1, B3 e C1.

**Issue C3 --- Implementar baseline de Centroid Tracking** -
**Objetivo:** produzir trajetórias de referência para a integração. -
**Tarefas:** distância máxima, criação/encerramento de tracks,
tratamento de ausências e exportação do Contrato B. - **Aceitação:**
cenários sintéticos e vídeo piloto processados; limitações
registradas. - **Dependências:** A1.

### Epic / marco técnico D --- Cinemática e decisão

**Issue D1 --- Validar geometria e homografia** - **Objetivo:**
documentar a validade da conversão pixel→metro. - **Tarefas:** definir
pontos de calibração, matriz, sistema de eixos e erro de reprojeção. -
**Aceitação:** pontos de validação e erro registrados; saída sinaliza
quando não há calibração válida. - **Dependências:** seleção da
câmera/vídeo.

**Issue D2 --- Implementar detecção de entradas e saídas de ROI** -
**Objetivo:** identificar ocupações da zona de conflito. - **Tarefas:**
regra geométrica, timestamps, transições, oclusões e eventos
incompletos. - **Aceitação:** testes sintéticos cobrem entrada, saída,
ausência e sobreposição. - **Dependências:** A1.

**Issue D3 --- Implementar PET com testes sintéticos** - **Objetivo:**
calcular PET somente para eventos elegíveis. - **Tarefas:** associação
de agentes, regra temporal, validade e limiar configurável. -
**Aceitação:** casos com PET conhecido retornam resultado esperado;
eventos ambíguos são sinalizados. - **Dependências:** D2.

**Issue D4 --- Formalizar espera e não-cedência** - **Objetivo:** evitar
indicadores ambíguos. - **Tarefas:** definir elegibilidade, início/fim,
denominadores e valores ausentes; documentar protocolo de anotação. -
**Aceitação:** dois anotadores conseguem aplicar as regras em uma
amostra piloto. - **Dependências:** A2.

**Issue D5 --- Calcular resumo de sessão e ICO baseline** -
**Objetivo:** agregar eventos auditáveis. - **Tarefas:** resumo de
sessão; normalização; pesos iniciais; tratamento de ausências; análise
de sensibilidade. - **Aceitação:** cada componente pode ser rastreado
aos eventos; resultados e limitações são exportados. - **Dependências:**
D3 e D4.

### Epic / marco técnico E --- Integração e demonstração

**Issue E1 --- Integrar pipeline ponta a ponta com vídeo piloto** -
**Objetivo:** conectar os módulos sem depender de mocks em todas as
etapas. - **Tarefas:** leitura do vídeo, detecção, rastreamento,
cinemática, eventos e exportação. - **Aceitação:** uma execução completa
produz artefatos verificáveis e erros são tratados sem encerrar
silenciosamente o processo. - **Dependências:** C2, C3 e módulos de
cinemática/decisão necessários.

**Issue E2 --- Produzir relatório de avaliação do baseline** -
**Objetivo:** preparar evidências para o Check-in 3. - **Tarefas:**
métricas, erros representativos, configurações, limitações e próximos
passos. - **Aceitação:** resultados reproduzíveis e distinção clara
entre teste sintético e validação em vídeo real. - **Dependências:** C1,
C2 e, conforme o escopo do check-in, E1.

### Como registrar as issues

Use o template já adotado pelo projeto. Se não houver template, cada
issue deve conter: - contexto e objetivo; - escopo incluído e
excluído; - tarefas verificáveis; - critérios de aceitação; -
dependências; - responsável; - milestone; - links para contrato, decisão
ou relatório relacionado.

Não marque uma issue como concluída apenas porque o código foi escrito.
A conclusão exige evidência dos critérios de aceitação.

------------------------------------------------------------------------

## 14. Ordem sugerida de execução

1.  Revisar o repositório e o Kanban existentes; evitar duplicações.
2.  Aprovar Contratos A/B/C e o protocolo de anotação.
3.  Preparar ambiente e testes mínimos.
4.  Selecionar um vídeo autorizado e verificar licença, qualidade e ROI.
5.  Anotar uma amostra pequena e validar a exportação.
6.  Implementar o avaliador de detecção com fixtures sintéticas.
7.  Executar o detector baseline e registrar métricas.
8.  Implementar Centroid Tracking e validar trajetórias.
9.  Testar geometria, ROIs e PET com dados sintéticos.
10. Integrar os módulos possíveis e produzir relatório do baseline.
11. Só então decidir quais arquiteturas alternativas merecem comparação.

Esta ordem é uma proposta de dependências técnicas; o time deve
conciliá-la com o milestone e o escopo oficial do próximo check-in.

------------------------------------------------------------------------

## 15. Definição de pronto (Definition of Done)

Uma issue técnica só deve ser marcada como concluída quando: - o
comportamento esperado está documentado; - os testes relevantes
passam; - o resultado foi revisado por outra pessoa quando a tarefa
afeta integração; - as mudanças respeitam os contratos ou incluem uma
atualização versionada deles; - não foram adicionados vídeos restritos,
segredos ou artefatos grandes ao Git; - os comandos e configurações
necessários estão registrados; - limitações e decisões metodológicas
estão documentadas; - os critérios de aceitação da própria issue foram
verificados.

------------------------------------------------------------------------

## 16. Links oficiais para consulta

Verificar sempre as instruções atuais de instalação e os termos
aplicáveis:

-   Python: https://www.python.org/downloads/
-   Ambientes virtuais Python:
    https://docs.python.org/3/library/venv.html
-   PyTorch --- instalação: https://pytorch.org/get-started/locally/
-   Ultralytics --- documentação: https://docs.ultralytics.com/
-   OpenCV --- documentação: https://docs.opencv.org/
-   CVAT --- documentação: https://docs.cvat.ai/
-   Label Studio --- documentação: https://labelstud.io/guide/
-   pytest: https://docs.pytest.org/
-   JSON Schema: https://json-schema.org/
-   MOTChallenge: https://motchallenge.net/
-   BDD100K: https://bdd-data.berkeley.edu/
-   JAAD: https://data.nvision2.eecs.yorku.ca/JAAD_dataset/

Os links acima são pontos de partida, não uma confirmação de que cada
conjunto contém cenas adequadas ao PET ou de que qualquer uso está
autorizado. Leia as licenças e condições de cada fonte antes de baixar,
anotar, compartilhar ou publicar dados.

------------------------------------------------------------------------

## 17. Entregáveis por marco

### Check-in 2 — apresentação imediata

Priorizar evidências conceituais e analíticas, sem afirmar que o pipeline de visão computacional já executa:
1. fundamentação do problema e definição dos indicadores;
2. vídeo proxy editado no CapCut, com caixas e ROIs explicitamente identificadas como elementos gráficos;
3. planilha sintética de trajetórias com tempos e posições conhecidos;
4. demonstração de regras determinísticas e cálculo de PET/indicadores com valores controlados;
5. plano de contingência para a pendência dos vídeos institucionais;
6. limitações, riscos e plano de implementação para o próximo marco.

### Check-in 3 — evidências de implementação e avaliação

Priorizar evidências pequenas e verificáveis: 1. contrato de dados
aprovado; 2. protocolo de anotação e amostra piloto; 3. vídeo autorizado
com metadados registrados; 4. avaliador de detecção com testes
sintéticos; 5. métricas iniciais do detector baseline; 6. demonstração
visual com ROIs e agentes anotados; 7. resultados dos testes sintéticos
de geometria/PET, se o módulo estiver implementado; 8. lista de
limitações, decisões pendentes e próximos experimentos.

**Mensagem central:** primeiro construir uma linha de base simples,
auditável e mensurável; depois comparar modelos mais sofisticados. A
contribuição científica depende tanto da validade da avaliação quanto da
arquitetura escolhida.
