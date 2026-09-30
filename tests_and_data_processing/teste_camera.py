# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:19:31 2026

@author: vinicius.ferreira
"""

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a camera")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar frame")
        break

    cv2.imshow("TESTE CAMERA", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()