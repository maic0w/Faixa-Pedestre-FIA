from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")  # Modelo pré-treinado tamanho nano

capture = cv2.VideoCapture(r"data\raw\BackVehcilesTraffic720p.mp4")
    
if not capture.isOpened():
    print("Erro ao abrir o vídeo. Verifique se o arquivo realmente está na pasta informada.")
    exit()

results = model.track(
    source=r"data\raw\BackVehcilesTraffic720p.mp4",  # Caminho do vídeo para rastreamento
    tracker="botsort.yaml",  # Uso do rastreador DeepSORT para rastrear objetos
    classes=[2, 3, 5, 7], # Filtra apenas veículos (classe 2: carro, classe 3: moto, classe 5: ônibus, classe 7: caminhão)
    line_width=1,  # Largura da linha para desenhar caixas delimitadoras
    conf=0.5,  # Confiança mínima para considerar uma detecção válida
    persist=True,  # Preserva os IDs de rastreamento entre frames
    save=True, # Salvar resultados de rastreamento em disco
    stream=True # Permite visualizar o vídeo com os resultados de rastreamento em tempo real
)

for result in results:
    frame_desenhado = result.plot()  # Plota os resultados de rastreamento no frame
    cv2.imshow("Rastreamento de Veículos", frame_desenhado)  # Exibe o frame com os resultados

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

capture.release()
cv2.destroyAllWindows()