import cv2
import numpy as np
import urllib.request
import mediapipe as mp

#ip
URL = "colocar la direccion IP de la camara del celular"

def rastrear_manos(url, titulo_ventana="Rastreo de manos"):
    mp_manos = mp.solutions.hands
    mp_dibujo = mp.solutions.drawing_utils
    mp_estilos = mp.solutions.drawing_styles

    manos = mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow(titulo_ventana, cv2.WINDOW_NORMAL)
    print("Conectando con el celular. Presiona 'q' para salir.")

    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)

            # MediaPipe trabaja en RGB, OpenCV entrega en BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = manos.process(frame_rgb)

            if resultado.multi_hand_landmarks:
                for puntos_mano in resultado.multi_hand_landmarks:
                    mp_dibujo.draw_landmarks(
                        frame,
                        puntos_mano,
                        mp_manos.HAND_CONNECTIONS,
                        mp_estilos.get_default_hand_landmarks_style(),
                        mp_estilos.get_default_hand_connections_style()
                    )

            cv2.imshow(titulo_ventana, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty(titulo_ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception as e:
            print(f"Error al obtener frame: {e}")
            break

    manos.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rastrear_manos(URL)