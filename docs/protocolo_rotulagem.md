# Protocolo Operacional de Rotulagem e Anotação (Ground Truth)

**Equipe:** Dupla 1 (João Pedro e Rafael)  
**Projeto:** PedestrianConflict-SE  
**Finalidade:** Padronizar as regras de anotação no CVAT / Label Studio para construção do *ground truth* piloto da disciplina de Fundamentos de IA.

---

## 1. Classes de Interesse e Mapeamento COCO

Para garantir interoperabilidade com modelos pré-treinados (YOLOv8 / YOLO11), as anotações adotam estritamente os IDs e nomes da taxonomia COCO:

| ID COCO | Nome da Classe | Descrição Operacional | Superclasse Interna |
|:---:|:---:|:---|:---:|
| `0` | `person` | Pedestres em pé, caminhando, correndo, em cadeiras de rodas ou parados na calçada/pista. | Pedestre |
| `1` | `bicycle` | Bicicletas (com ou sem ciclista visível como parte da montaria). | Não-motorizado |
| `2` | `car` | Veículos de passeio, sedãs, SUVs, táxis, viaturas e caminhonetes leves. | Veículo Motorizado |
| `3` | `motorcycle` | Motocicletas, motonetas e ciclomotores (incluindo o condutor montado). | Veículo Motorizado |
| `5` | `bus` | Ônibus urbanos, micro-ônibus e ônibus de turismo. | Veículo Motorizado |
| `7` | `truck` | Caminhões, carretas, caminhões de carga e veículos pesados de serviço. | Veículo Motorizado |

> **Nota de Anotação:** Pedestres carregando bolsas, mochilas ou guarda-chuvas devem ter a caixa delimitadora cobrindo o corpo do pedestre; acessórios pequenos não devem estender a caixa excessivamente. Para ciclistas e motociclistas montados, anote o conjunto veículo + condutor sob a respectiva classe (`bicycle` ou `motorcycle`).

---

## 2. Padrão de Formato de Exportação (YOLO `.txt`)

Cada imagem anotada `frame_000120.jpg` deve possuir o respectivo arquivo de texto `frame_000120.txt` no mesmo diretório ou em `labels/`.

### Estrutura por linha:
```text
<class_id> <x_center> <y_center> <width> <height>
```
* **Normalização:** Todos os valores numéricos devem estar no intervalo $[0.0, 1.0]$, divididos pela largura ($W$) e altura ($H$) do quadro.
* **Exemplo de anotação de 1 pedestre e 1 carro:**
  ```text
  0 0.234125 0.654312 0.052140 0.184320
  2 0.612500 0.540000 0.220000 0.160000
  ```

---

## 3. Diretrizes de Qualidade e Casos Especiais

### 3.1 Ponto de Contato com o Solo
* Em câmeras oblíquas, a base inferior da caixa delimitadora ($y_2$) define o ponto de contato com o piso da via para a homografia e para os polígonos Shapely da faixa de pedestres.
* Certifique-se de que a borda inferior da caixa coincide com o ponto em que os pés do pedestre ou as rodas do veículo tocam o solo.

### 3.2 Oclusões Parciais e Totais
* **Oclusão parcial ($< 70\%$ visível):** Anotar a caixa delimitadora inferindo o tamanho completo estimado do agente.
* **Oclusão severa ($> 70\%$ encoberto):** NÃO anotar se não for possível inferir confiavelmente a silhueta ou os limites do agente.
* **Oclusão temporária em vídeo:** Se o anotador usar CVAT com *tracking*, manter o mesmo `track_id` assim que o objeto reemergir.

### 3.3 Truncamento na Borda da Imagem
* Agentes parcialmente cortados pelas bordas do quadro devem ser anotados cobrindo estritamente a porção visível dentro dos limites da imagem $[0, W] \times [0, H]$.

### 3.4 Resolução Mínima
* Agentes com altura inferior a 15 pixels (muito distantes no horizonte da via) não devem ser anotados para evitar ruído no cálculo de precisão/revocação do baseline.

---

## 4. Procedimento de Calibração Inter-Anotadores (Piloto)

1. **Lote Piloto:** Extrair 150 quadros do vídeo de teste via `sample_extractor.py`.
2. **Duplo-Cego Inicial:** João Pedro e Rafael anotam independentemente os primeiros 30 quadros.
3. **Avaliação Cruzada:** Executar `src/evaluation/evaluator.py` comparando as duas anotações para medir a concordância ($IoU \ge 0.50$, $F_1 \ge 0.85$).
4. **Resolução de Conflitos:** Ajustar discrepâncias de critério (ex: limites de calçada, veículos estacionados vs. em movimento).
5. **Anotação Restante:** Dividir os quadros subsequentes (120 quadros) entre os integrantes.
