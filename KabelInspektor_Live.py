#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KabelInspektor Live — Webcam-basierte Erkennung von Kabel-/Leitungsschäden
Autor: (Ihr Name)
Version: 1.0

Beschreibung
------------
Dieses Tool analysiert einen Live-Video-Stream (Webcam) und versucht,
auffällige Stellen an einem Kabel/ einer Leitung zu detektieren.
Es nutzt klassische Computer-Vision-Verfahren (OpenCV), ist offline lauffähig
und bietet Regler (Trackbars) zur Feinjustierung in Echtzeit.

Wichtige Hinweise
-----------------
- Prototyp für die visuelle Qualitätsprüfung. Nicht als einziges
  sicherheitsrelevantes Prüfmittel verwenden.
- Am besten funktioniert es, wenn das Kabel im Bild dominiert und vor
  einem ruhigen Hintergrund geführt wird (Kontrast!).
- Lichtverhältnisse möglichst konstant halten.

Funktionen (Kurzüberblick)
--------------------------
1) Vorverarbeitung mit CLAHE (Kontrastverbesserung)
2) Kanten-/Konturerkennung zum Finden der "Haupt-Leitung"
3) Ausrichtung des ROI entlang der Hauptachse (Rotation)
4) Breitenprofil-Analyse: Lokale Abweichungen (Ausdünnung/Beule/Bruch)
5) Helligkeitsanomalien: sehr helle Flecken (z.B. freiliegendes Metall)
6) Live-Overlay, Regler für Parameter, Snapshot/Logging per Tastendruck

Tastatursteuerung
-----------------
[q]  : Beenden
[s]  : Snapshot (PNG) speichern
[l]  : CSV-Log mit aktuellen Kennzahlen anhängen
[ ]  : Leertaste: Pausieren/Fortsetzen

Abhängigkeiten
--------------
- Python 3.8+
- OpenCV (cv2) 4.x
- NumPy
Installieren (Beispiel):
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import time
import csv
from datetime import datetime
from pathlib import Path

# =============== Hilfsfunktionen ============================================

def create_trackbar_window():
    cv2.namedWindow("Regler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Regler", 430, 420)
    # Canny
    cv2.createTrackbar("Canny Low", "Regler", 60, 255, lambda v: None)
    cv2.createTrackbar("Canny High", "Regler", 140, 255, lambda v: None)
    # Morphologie
    cv2.createTrackbar("Morph-Kernel", "Regler", 3, 15, lambda v: None)
    cv2.createTrackbar("Min Kontur-Fläche", "Regler", 8000, 100000, lambda v: None)
    # Breitenanalyse
    cv2.createTrackbar("Toleranz %", "Regler", 25, 80, lambda v: None)  # prozentuale Abweichung vom Median
    cv2.createTrackbar("Min Defekt-Länge", "Regler", 25, 300, lambda v: None)  # in Pixel (Spalten im rotierten ROI)
    # Helligkeitsanomalie
    cv2.createTrackbar("Helligkeit K", "Regler", 18, 60, lambda v: None)  # Schwelle = Mittelwert + K/10*Std
    cv2.createTrackbar("Min Hell-Fläche", "Regler", 50, 2000, lambda v: None)
    cv2.createTrackbar("Max Hell-Fläche", "Regler", 1000, 10000, lambda v: None)
    # CLAHE
    cv2.createTrackbar("CLAHE Clip*10", "Regler", 20, 60, lambda v: None) # 2.0 … 6.0
    cv2.createTrackbar("CLAHE Tiles", "Regler", 8, 16, lambda v: None)


def get_trackbar_values():
    low = cv2.getTrackbarPos("Canny Low", "Regler")
    high = cv2.getTrackbarPos("Canny High", "Regler")
    k = max(1, cv2.getTrackbarPos("Morph-Kernel", "Regler"))
    min_area = cv2.getTrackbarPos("Min Kontur-Fläche", "Regler")
    tol = max(5, cv2.getTrackbarPos("Toleranz %", "Regler"))
    min_def_len = max(5, cv2.getTrackbarPos("Min Defekt-Länge", "Regler"))
    bright_k = cv2.getTrackbarPos("Helligkeit K", "Regler") / 10.0
    min_bright_area = cv2.getTrackbarPos("Min Hell-Fläche", "Regler")
    max_bright_area = max(min_bright_area+10, cv2.getTrackbarPos("Max Hell-Fläche", "Regler"))
    clip = max(1, cv2.getTrackbarPos("CLAHE Clip*10", "Regler")) / 10.0
    tiles = max(2, cv2.getTrackbarPos("CLAHE Tiles", "Regler"))
    return {
        "canny_low": low, "canny_high": high, "kernel": k, "min_area": min_area,
        "tol_pct": tol, "min_def_len": min_def_len, "bright_k": bright_k,
        "min_bright_area": min_bright_area, "max_bright_area": max_bright_area,
        "clahe_clip": clip, "clahe_tiles": tiles
    }


def clahe_enhance_bgr(frame, clip=2.0, tiles=8):
    # CLAHE im LAB-L Kanal
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def find_main_contour(mask, min_area=8000):
    # Größte langgestreckte Kontur finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / (min(w, h) + 1e-6)
        score = area * aspect  # Fläche * Schlankheitsgrad
        if score > best_score:
            best_score = score
            best = cnt
    return best


def rotate_image_and_mask(image, mask, rect):
    # ROI entlang der Hauptachse ausrichten
    (cx, cy), (w, h), angle = rect
    # OpenCV Winkel: Rechteck ist gegen Uhrzeigersinn gedreht;
    # für horizontale Ausrichtung drehen wir um "angle"
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    return rotated_img, rotated_mask, M


def invert_affine(M):
    # 2x3 Affinmatrix in 3x3 homogen erweitern und invertieren
    A = np.vstack([M, [0,0,1]])
    A_inv = np.linalg.inv(A)
    return A_inv[:2,:]


def columns_width_profile(rot_mask):
    h, w = rot_mask.shape[:2]
    widths = np.zeros(w, dtype=np.int32)
    for x in range(w):
        col = rot_mask[:, x]
        ys = np.flatnonzero(col)
        if ys.size > 0:
            widths[x] = ys.max() - ys.min() + 1
    return widths


def group_runs(mask_1d):
    # Gruppen von True-Bereichen in 1D Maske
    runs = []
    start = None
    for i, val in enumerate(mask_1d):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i-1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)-1))
    return runs


def bright_spots(rot_gray, rot_mask, thresh_k=1.8, min_area=50, max_area=2000):
    # Helle Flecken innerhalb der Maske (Mittel + k*Std)
    vals = rot_gray[rot_mask>0]
    if vals.size == 0:
        return []
    m = float(np.mean(vals))
    s = float(np.std(vals)) + 1e-6
    T = m + thresh_k * s
    bin_img = cv2.inRange(rot_gray, int(T), 255)
    bin_img = cv2.bitwise_and(bin_img, rot_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,w,h))
    return boxes


def draw_rotated_box(dst, box, Minv, color=(0,0,255), thickness=2):
    # box in rotierten Koordinaten -> zurücktransformieren
    x,y,w,h = box
    pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
    ones = np.ones((4,1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    mapped = (Minv @ pts_h.T).T
    mapped = mapped.astype(np.int32)
    cv2.polylines(dst, [mapped], isClosed=True, color=color, thickness=thickness)


def put_multi_line_text(img, text, org=(10,20), line_height=18, color=(0,255,0)):
    x, y = org
    for i, line in enumerate(text.splitlines()):
        cv2.putText(img, line, (x, y + i*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# =============== Hauptprogramm ==============================================

def main():
    print("Starte Kamera…")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # ggf. Index anpassen
    if not cap.isOpened():
        print("Fehler: Keine Kamera gefunden.")
        return

    # Optionale Parameter
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    create_trackbar_window()

    out_dir = Path("kabelinspektor_output")
    out_dir.mkdir(exist_ok=True)

    paused = False
    prev_time = time.time()
    fps = 0.0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Frame konnte nicht gelesen werden.")
                break
        else:
            # Wenn pausiert, Bild nicht erneuern.
            pass

        params = get_trackbar_values()

        # 1) Vorverarbeitung
        frame_enh = clahe_enhance_bgr(frame, clip=params["clahe_clip"], tiles=params["clahe_tiles"])
        gray = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2GRAY)

        # 2) Kanten + Morphologie, um eine solide Maske zu erhalten
        edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])
        k = max(1, params["kernel"] | 1)  # ungerade
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dil = cv2.dilate(closed, kernel, iterations=1)
        # Konturenmaske erstellen (gefüllt)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # 3) Haupt-Kontur (Kabel) auswählen
        main_cnt = find_main_contour(mask, min_area=params["min_area"])
        display = frame.copy()
        if main_cnt is None:
            put_multi_line_text(display, "Hinweis: Keine geeignete Kontur gefunden.\n"
                                         "Tipp: Hintergrund vereinfachen, Regler anpassen.", (10,20))
            cv2.imshow("Live", display)
            cv2.imshow("Kanten", edges)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            continue

        rect = cv2.minAreaRect(main_cnt)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(display, [box], True, (0,255,255), 2)

        # 4) Rotation auf Hauptachse
        rot_img, rot_mask, M = rotate_image_and_mask(frame_enh, (mask>0).astype(np.uint8)*255, rect)
        Minv = invert_affine(M)
        rot_gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)

        # Optional: ROI Einschränken auf boundingRect der rotierten Maske
        x,y,w,h = cv2.boundingRect(rot_mask)
        roi_mask = rot_mask[y:y+h, x:x+w]
        roi_gray = rot_gray[y:y+h, x:x+w]
        # Sicherheitscheck
        if roi_mask.size == 0 or np.count_nonzero(roi_mask)==0:
            cv2.imshow("Live", display)
            cv2.imshow("Kanten", edges)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            continue

        # 5) Breitenprofil
        widths = columns_width_profile(roi_mask)
        nonzero = widths[widths>0]
        if nonzero.size > 0:
            med_width = np.median(nonzero)
        else:
            med_width = 0

        tol = params["tol_pct"] / 100.0
        lower = med_width * (1.0 - tol)
        upper = med_width * (1.0 + tol)
        deviants = (widths>0) & ((widths<lower) | (widths>upper))

        runs = group_runs(deviants)
        defect_boxes_rot = []
        for (xs, xe) in runs:
            if (xe - xs + 1) >= params["min_def_len"]:
                # Höhe der Maske in diesem Bereich
                col_slice = roi_mask[:, xs:xe+1]
                ys = np.flatnonzero(np.any(col_slice>0, axis=1))
                if ys.size == 0:
                    continue
                y_top, y_bottom = int(ys.min()), int(ys.max())
                defect_boxes_rot.append((x+xs, y+y_top, xe-xs+1, y_bottom - y_top + 1))

        # 6) Helligkeitsanomalien
        bright_boxes_rot = []
        try:
            bxs = bright_spots(roi_gray, roi_mask, thresh_k=params["bright_k"],
                               min_area=params["min_bright_area"],
                               max_area=params["max_bright_area"])
            # In globale (rotierte Vollbild-)Koordinaten umrechnen
            bright_boxes_rot = [(x+bx, y+by, bw, bh) for (bx,by,bw,bh) in bxs]
        except Exception as e:
            # robust bleiben, falls Statistik fehlschlägt
            pass

        # 7) Zurück in Original: Boxen zeichnen
        for b in defect_boxes_rot:
            draw_rotated_box(display, b, Minv, color=(0,0,255), thickness=2)
        for b in bright_boxes_rot:
            draw_rotated_box(display, b, Minv, color=(0,255,0), thickness=2)

        # FPS
        now = time.time()
        fps = 0.9*fps + 0.1*(1.0/(now - prev_time + 1e-6))
        prev_time = now

        # 8) Overlay-Infos
        info = [
            f"FPS: {fps:5.1f}",
            f"Medianbreite: {med_width:.1f} px",
            f"Defekt-Segmente: {len(defect_boxes_rot)}",
            f"Helle Spots: {len(bright_boxes_rot)}",
            "Tasten: [q]=Quit  [s]=Snapshot  [l]=CSV-Log  [Space]=Pause"
        ]
        put_multi_line_text(display, "\n".join(info), (10,20), color=(50,230,50))

        cv2.imshow("Live", display)
        cv2.imshow("Kanten", edges)
        cv2.imshow("ROI (rotiert)", cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = out_dir / f"snapshot_{ts}.png"
            cv2.imwrite(str(fn), display)
            print(f"Snapshot gespeichert: {fn}")
        elif key == ord('l'):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_path = out_dir / "kabelinspektor_log.csv"
            newfile = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                wr = csv.writer(f, delimiter=";")
                if newfile:
                    wr.writerow(["timestamp","fps","median_width_px","defect_segments","bright_spots",
                                 "canny_low","canny_high","kernel","min_area","tol_pct","min_def_len",
                                 "bright_k","min_bright_area","max_bright_area","clahe_clip","clahe_tiles"])
                wr.writerow([ts, f"{fps:.2f}", f"{med_width:.1f}", len(defect_boxes_rot), len(bright_boxes_rot),
                             params["canny_low"], params["canny_high"], params["kernel"], params["min_area"],
                             params["tol_pct"], params["min_def_len"], params["bright_k"],
                             params["min_bright_area"], params["max_bright_area"], params["clahe_clip"], params["clahe_tiles"]])
            print(f"CSV-Log aktualisiert: {csv_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
