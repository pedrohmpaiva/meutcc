import cv2
import dlib
import os
import glob

#Indicação dos caminhos onde estão os arquivos .svm
detectorMascara = dlib.simple_object_detector("fotos/treinamento/detector_mascaras5.svm")
detectorFace = dlib.simple_object_detector("fotos/treinamentoS/semmascara.svm")

#Este for percorre toda a pasta em busca dos arquivos
#Aqui basicamente ele pesquisa a imagem, o OpenCV as lê e atribui a uma variável
#E por fim as atribui aos respectivos detectores
for imagem in glob.glob(os.path.join("mascara teste", "*.jpg")):
    pic = cv2.imread(imagem)
    pic2 = cv2.imread(imagem)
    objetosDetectados = detectorMascara(pic, 3)
    objetosDetectados2 = detectorFace(pic2, 3)

    # Estes dois for's pegam as posições que os arquivos svm forneceram das respectivas imagens e faz um retangulo em
    # volta e também exibe um texto. Como ainda não descobri uma forma de fazer uma condição para avaliar qual o
    # melhor detector de acordo com o cenário, os dois executam e se for detectado algo, ele faz o retorno.

    for i in objetosDetectados:
        e, t, d, b = (int(i.left()), int(i.top()), int(i.right()), int(i.bottom()))
        cv2.rectangle(pic, (e, t), (d, b), (0, 255, 0), 2)
        cv2.putText(pic, 'Possivel Mascara ' + str(i), (e, b + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow("Detector de máscaras", pic)

        cv2.waitKey(0)

    for i in objetosDetectados2:
        e, t, d, b = (int(i.left()), int(i.top()), int(i.right()), int(i.bottom()))
        cv2.rectangle(pic, (e, t), (d, b), (0, 0, 255), 2)
        cv2.putText(pic, 'Pessoa sem mascara ' + str(i), (e, b + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow("Detector de máscaras", pic)

        cv2.waitKey(0)


cv2.destroyAllWindows()