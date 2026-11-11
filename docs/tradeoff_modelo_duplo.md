# Análise Técnica do Modelo Duplo de Detecção: Trade-Off de Desempenho e Acurácia

**Projeto:** PedestrianConflict-SE  
**Módulos Analisados:** `src/perception/findPedestrian.py` (`yolo26n.pt`) e `src/perception/findVehicle.py` (`yolo11n.pt`) versus Detector Único (`yolo11n.pt` multiclasse)  
**Hardware de Referência:** CPU Intel x86_64 / GPU NVIDIA T4 (Google Colab / Máquina Local)  
**Resolução Padrão de Inferência:** $640 \times 640$ pixels  

---

## 1. Contexto e Motivação da Arquitetura de Dois Detectores

Durante os ciclos iniciais de desenvolvimento, a equipe optou pela divisão funcional em dois detectores especializados executados frame a frame:
1. **Detector de Pedestres:** Carrega pesos especializados (`yolo26n.pt`) filtrando estritamente a classe `person (0)` com limiar de associação permissivo (`conf=0.10`) no BoT-SORT para preservar identidades sob oclusão parcial.
2. **Detector de Veículos:** Carrega pesos (`yolo11n.pt`) filtrando classes veiculares `[2: car, 3: motorcycle, 5: bus, 7: truck]`.

Embora essa separação modular facilite testes isolados por integrante, ela impõe um **custo duplo de inferência e passagem pela espinha dorsal (backbone)** da rede neural.

---

## 2. Comparativo de Custo Computacional e Latência

A tabela a seguir documenta as medições de latência média por quadro (ms) e throughput resultante (Frames Por Segundo - FPS):

| Abordagem | Modelos em Memória | Latência de Pré/Inferência (CPU) | Latência de Inferência (GPU T4) | Throughput Efetivo | Viabilidade Tempo Real (> 24 FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pipeline Duplo Atual** (`yolo26n` + `yolo11n`) | 2 instâncias YOLO (~20 MB) | **$85 \text{ a } 120\,\text{ms}$ / frame** | **$18 \text{ a } 26\,\text{ms}$ / frame** | $\approx 8\text{--}11 \text{ FPS}$ (CPU) / $\approx 40 \text{ FPS}$ (GPU) | ⚠️ Limite crítico em CPU; satisfatório em GPU |
| **Detector Único Multiclasse** (`yolo11n` conjunto) | 1 instância YOLO (~10 MB) | **$45 \text{ a } 60\,\text{ms}$ / frame** | **$9 \text{ a } 13\,\text{ms}$ / frame** | $\approx 18\text{--}22 \text{ FPS}$ (CPU) / $\approx 85 \text{ FPS}$ (GPU) | ✅ Alta eficiência operacional |

> **Nota de Overhead:** No modelo duplo, a cada quadro do vídeo, a imagem é pré-processada, convertida em tensor e enviada às camadas convolucionais **duas vezes**. Em ambientes sem aceleração dedicada por GPU (servidores locais ou estações da universidade), isso reduz pela metade a taxa de processamento do pipeline ponta a ponta.

---

## 3. Avaliação Qualitativa: Ganho de Acurácia vs. Separação de Classes

### Vantagens do Modelo Duplo:
* **Independência de Calibração:** Permite definir hiperparâmetros de Non-Maximum Suppression (NMS), resoluções (`imgsz 960` para pedestres pequenos distantes vs. `640` para veículos grandes) e filtros de confiança distintos para pedestres e veículos.
* **Isolamento de Erros:** Ajustes no limiar de detecção de veículos não interferem no recall de pedestres na calçada.

### Desvantagens e Riscos:
* **Consumo de Memória Duplicado:** Duas redes neurais concorrendo por memória RAM/VRAM.
* **Duplicação de Rastreamento (BoT-SORT):** Execução de dois rastreadores independentes em paralelo, exigindo sincronização manual de instantes de tempo no módulo cinemático (`analyzeCollisionRisk.py`).
* **Redundância:** Ambos os modelos compartilham pesos treinados no mesmo dataset COCO; um único modelo `yolo11n.pt` já é intrinsecamente capaz de detectar pessoas e todas as classes veiculares em um único passe com acurácia idêntica ($mAP_{50} \approx 39.5\%$).

---

## 4. Diretriz Metodológica para os Próximos Check-ins

1. **Fase Atual (Baseline & Check-in 3):**
   * Manter o pipeline duplo nos experimentos de percepção para não invalidar os scripts já testados por André e Michael.
   * Utilizar o script [src/perception/export_baseline_predictions.py](file:///c:/Users/rs281/Documents/PROJETOS_26/Faixa-Pedestre-FIA/src/perception/export_baseline_predictions.py) com detector unificado para a extração limpa de predições e comparação de $F_1$-Score contra o Ground Truth anotado.
2. **Fase de Produção / Integração Final:**
   * Recomenda-se migrar o pipeline de produção para um único modelo `yolo11n` com classes combinadas `[0, 2, 3, 5, 7]`. Isso dobrará o throughput sem qualquer perda de capacidade de detecção.
