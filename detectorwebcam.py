import dlib
import cv2
import sys


quadros = 40 #taxa de quadros
captura = cv2.VideoCapture(0) #referencia a câmera que será usada
contadorQuadros = 0 # Essa variável vai ajudar nas divisão dos quadros capturados. Ou seja, o processamento
                    # só irá ocorrer de 30 em 30 quadros e não para cada quadro capturado

detector = dlib.simple_object_detector("fotos/treinamento/detector_mascaras5.svm") #Caminho dos detectores
detectorFace = dlib.simple_object_detector("fotos/treinamentoS/semmascara.svm")


while captura.isOpened():
    conectado, frame = captura.read()
    contadorQuadros += 1
    if contadorQuadros % quadros == 0:
        objetosDetectados = detector(frame, 1) # Aqui o que foi capturado pela webcam é atribuido a ser processado pelos
                                                # detectores
        objetosDetectados2 = detectorFace(frame, 1)

        # Aqui ocorre o desenho do quadrado no objeto detectado
        for i in objetosDetectados:
            objetosDetectados = detector(frame, 1)
            e, t, d, b = (int(i.left()), int(i.top()), int(i.right()), int(i.bottom()))
            cv2.rectangle(frame, (e, t), (d, b), (0, 0, 255), 2)
            cv2.putText(frame, 'Possivel Mascara ' + str(i), (e, b + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.imshow("Detector de máscaras", frame)
            cv2.waitKey(0)
        for i in objetosDetectados2:
            objetosDetectados2 = detectorFace(frame, 1)
            e, t, d, b = (int(i.left()), int(i.top()), int(i.right()), int(i.bottom()))
            cv2.rectangle(frame, (e, t), (d, b), (0, 0, 255), 2)
            cv2.putText(frame, 'Sem mascara ' + str(i), (e, b + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 0),
                        1,
                        cv2.LINE_AA)
            cv2.imshow("Detector de máscaras", frame)
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == 27: # Aqui é passada a tecla ESC para parar o código
            break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)
