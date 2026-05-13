# Visualisierung 
Du bist ein erfahrener Python-Entwickler mit Expertise in Gradio-UIs und Robotik-Softwarearchitektur. Ich benötige eine modulare Gradio-Visualisierung für ein Robohund-Projekt, die in mein bestehendes Projekt integriert wird.

**Layout** (exakt wie im angehängten Bild `WhatsApp Image 2026-05-13 at 20.47.49.jpeg`)

Zwei Zeilen:
- **Zeile 1 (groß):** `Kamera Bild` (lila) | `Mapping` (gelb) — beide gleichgroß, nehmen die volle Breite ein
- **Zeile 2 (kleiner, gleichbreit):** `Trash Bilder Liste` (dunkelblau) | `Log-Daten` (türkis) | `Steuerung` (braun)

---

**Panel-Anforderungen:**

**🟣 Kamera Bild (lila)**
- Live-Feed des Robohund-Kamerabilds
- Aktueller Zeitstempel, FPS-Anzeige

**🟡 Mapping (gelb)**
- Karte die der Robohund in Echtzeit aufzeichnet (z.B. occupancy grid oder Pfadverlauf)
- Aktuelle Position des Robohunds eingezeichnet

**🔵 Trash Bilder Liste (dunkelblau)**
- Galerie/Liste der erkannten Müll-Objekte mit zugehörigem Kamerabild
- Anzeige von Klasse/Label, Confidence-Score, Zeitstempel und GPS-/Kartenposition pro Erkennung
- Scrollbare Liste wenn mehrere Erkennungen vorliegen

**🩵 Log-Daten (türkis)**
- Scrollbares Echtzeit-Log aller relevanten Systemereignisse
- Einträge mit Zeitstempel, Log-Level (INFO, WARN, ERROR), Quelle (Modul) und Nachricht
- Farbkodierung nach Log-Level
- Anzahl Einträge und Filtermöglichkeit nach Level

**🟤 Steuerung (braun)**
- Button: **Stop** — hält den Robohund sofort an
- Button: **Start** — startet die autonome Mission
- Button: **Zurück zum Start** — fährt zur Ausgangsposition zurück
- Button: **Manuelle Übernahme** — wechselt in manuellen Modus
- Button: **Karte speichern** — exportiert die aktuelle Map
- Statusanzeige: aktueller Modus, Batteriestand, Verbindungsstatus
- Erweiterbar — implementiere die Buttons als konfigurierbare Liste, damit weitere leicht hinzugefügt werden können

---

**Technische Anforderungen:**

- Vollständig **modularer Aufbau**: Jedes Panel ist eine eigene Python-Klasse oder Funktion in einer separaten Datei (z.B. `panels/camera.py`, `panels/map.py`, etc.) mit einem klar definierten Interface
- Hauptdatei `visualization.py` (oder ähnlich) importiert und assembliert alle Module
- Integration in bestehendes Projekt: keine hartcodierten Pfade, Konfiguration über eine zentrale `config.py` oder Umgebungsvariablen
- Datenübergabe über klar definierte Callbacks/Queues, sodass die Visualisierung von der Robotik-Logik entkoppelt bleibt
- Gradio `Blocks`-Layout mit `gr.Row` und `gr.Column` für exaktes Layout-Matching
- Farb-Theming der Panel-Borders/Header passend zu den vorgegebenen Farben (lila, gelb, dunkelblau, türkis, braun)
- Kommentierter, produktionsreifer Code mit Docstrings
- Dummy-Daten/Mock-Callbacks als Platzhalter wo echte Robotik-Daten eingespeist werden müssen, klar markiert mit `# TODO: connect to real data source`

Liefere die vollständige Dateistruktur, alle Dateien mit vollständigem Code und eine kurze Erklärung wie die Module in das bestehende Projekt eingebunden werden.

## Auswertung

[Ausführliche Beschreibung der Metric](prompt_metricen.md)

| Metric                              | Score         |
|-------------------------------------|---------------|
| **Tool used**                       | Codex ChatGPT |
| **Error Rate (0 - 4)**              | 4             |
| **Code Quality (0 - 4)**            | 3             |
| **Discrepancy from Prompt (0 - 4)** | 3             |
| **Notes**                           | -             |