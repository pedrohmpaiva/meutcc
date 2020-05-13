import dlib


opcoes = dlib.simple_object_detector_training_options() # Este parametro permite que eu dê opções extras para o
                                                        # treinamento do algoritmo svm
opcoes.add_left_right_image_flips = True # Este primeiro comando serve para mudar o angulo das imagens de aprendizado
                                         # automaticamente
opcoes.C = 7 # Este parâmetro é o "custo" do algoritmo. Quanto maior o número melhor o resultado. Porém será menos eficaz
             # em prever novos padrões


#Por fim, aqui eu passo o xml com as posições das imagens e ele irá me retornar o algoritmo pronto
dlib.train_simple_object_detector("fotos/treinamento/apenas_mascara.xml", "fotos/treinamento/detector_mascaras5.svm", opcoes)

