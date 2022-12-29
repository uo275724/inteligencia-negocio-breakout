# Entrega de Teoría 1 & 2
---
### Asignatura: Inteligencia de Negocio
### Autores:
- Alonso Menéndez González (UO275724)
- Diego García Vega (UO276409)
### Fecha: 18/11/2022 - 31/12/2022
### Título: Reinforcement Learning Atari Breakout
---

> En este proyecto se trabajará con Aprendizaje por Refuerzo (Reinforcement Learning) para entrenar un modelo capaz de ganar al clásico juego de Atari, Breakout.

# Ramas del proyecto
---
> GPU:
En esta rama se encuentra el modelo con la posibilidad de ser entrenado por GPU, que en caso de no encontrarla, este usará CPU. Debido a que las clases están hechas para GPU, la inferencia se realiza con ella, lo cual no es muy adecuado para un modelo tan simple, esto se solventa en otras ramas.

> Modelo1:
En esta rama se encuentra el modelo con las especificaciones congeladas en las que fue entrenado. Para en caso de que alguien desee repetir el entrenamiento pueda hacerlo, este tomó alrededor de unas 3 horas en una RTX 3080 con 12GB de memoria.

> OpenCV:
En esta rama se usará el modelo desarrollado en las ramas anteriores, ya entrenado, y se desacoplará del juego haciendo que coja las coordenadas de la raqueta mediante visión por computador a través de imágenes de este.

# Archivos del proyecto
---
> breakout_original.py:
En este fichero se encuentra el juego original, sin modificar creado por:
https://github.com/MatthewTamYT/Breakout

> breakout_human.py:
En este archivo se encuentra el juego modificado para que se adapte a las clases de Python que necestiamos, el funcionamiento es igual al anterior.

> breakout_IA.py:
Este archivo es tiene la misma estructura que el anterior, pero está orientado a ser operado por la inteligencia artificial, de forma que la velocidad de juego está aumentada, empieza de manera automática y al perder la partida vuelve a empezar otra nueva.

> agent.py:
En este archivo se define la clase agente, encargada de entrenar el modelo, obtener recompensas y transmitir las salidas de la red neuronal al videojuego.

> model.py:
En este archivo se definen las clases del modelo así como la red neuronal que se usará para entrenar.

> production.py
Mediante este archivo se puede ejecutar el modelo ya entrenado, en función de la rama en la que se encuentre, este usará visión por computador (OpenCV) o estará incrustado en el juego (GPU).

> cv.py
En este archivo se define la función que se usará para visión por computador además, de si este es ejecutado se observará la salida con una imagen de prueba.

> helper.py
En este archivo se encuentra una función auxiliar, que servirá para dibujar los datos del entrenamiento del modelo, la puntuación y la tendencia de este.

![Computer Vision](https://github.com/uo275724/inteligencia-negocio-breakout/blob/OpenCV/ComputerVision.png)

![Training gif](https://github.com/uo275724/inteligencia-negocio-breakout/blob/OpenCV/Training.gif)