import cv2
from ultralytics import YOLO

#YOLO DEFINIÇÃO:
 #YOLO( You Only Look Once ), é um modelo de detecção de objeto popular conhecido por sua velocidade e precisão, primeiramente introduzido por Joseph Redmon em 2016, com sua iteração mais recente sendo o YOLO v7.

 #O algorítimo YOLO recebe uma imagem como entrada e usa uma rede neural profunda para detectar objetos na imagem. O modelo divide a imagem de entrada numa grade S x S, com uma célula de grade ficando responsável pela detecção do objeto se o centro do mesmo cair nela.

 #Cada célula de grade vai prever caixas delimitadoras e placares de confiança para cada uma dessas caixas, com esses placares informando o quão confiante o modelo está de que a caixa contém um objeto e o quão preciso o modelo acha que a caixa prevista é.

 #Uma técnica chave do modelo YOLO é a supressão não máxima, que é usada para identificar e remover caixas redudantes ou incorretas e só ter uma caixa delimitadora para cada objeto na imagem.

# Carregar o modelo YOLO
yolo = YOLO('yolov8s.pt')

# Carrega o vídeo pré baixado
videoCap = cv2.VideoCapture('elefant_1280p.mp4')

# Carrega a webcam
#videoCap = cv2.VideoCapture(0)

# OBS: Para trocar entre video pré-gravado e webcam, só comentar um e tirar o comentário do outro


# Função para obter cores aleatórias de classe para a detecção de objeto
def obterCores(cls_num):
    cores_base = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    cor_index = cls_num % len(cores_base)
    incrementos = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    cor = [cores_base[cor_index][i] + incrementos[cor_index][i] * 
    (cls_num // len(cores_base)) % 256 for i in range(3)]
    return tuple(cor)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)
    
    for result in results:
        # obtem os nomes das classes
        classes_names = result.names

    # itera cada caixa
    for box in result.boxes:
            # cheque se o indice de confiança é maior que 60%
         if box.conf[0] > 0.6:
            # Obtem coordenadas
            [x1, y1, x2, y2] = box.xyxy[0]
            # converte para int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # obtem a classe
            cls = int(box.cls[0])

            # obtem o nome da classe
            class_name = classes_names[cls]

            # obtem a respectiva cor
            colour = obterCores(cls)

            # desenha a caixa
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # coloca o nome da classe e o indice de confiança na imagem
            cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # Mostra a imagem em uma tela a parte e programa a mesma para parar se apertado a tecla Q
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# Fecha a janela e termina o programa quando sai do loop
videoCap.release()
cv2.destroyAllWindows()