# Detektions-Validierung – UI (Schritt 1)

**Datum:** 2026-06-18
**Status:** Freigegeben (Brainstorming abgeschlossen)
**Scope dieses Schritts:** Nur die Oberfläche. Keine Festplatten-Persistenz.

## Ziel

Eine Oberfläche, mit der erkannte Müll-Detektionen manuell bewertet werden können:
für jede Detektion entscheidet ein Mensch **Müll ja / nein / unsicher**. Daraus ergibt
sich eine **Precision**-Kennzahl (wie viele Detektionen waren echte Treffer).

Dieser Schritt liefert **ausschließlich die UI** im bestehenden Gradio-Dashboard.
Das dauerhafte Speichern der Detektionen und Verdikte (`DetectionStore`, PNG/JSON auf
Platte, Export) ist bewusst **nicht** Teil dieses Schritts und folgt später.

## Nicht-Ziele (YAGNI)

- Keine Persistenz auf Festplatte (Bilder/Verdikte nur im Arbeitsspeicher).
- Kein echter Export (Button ist Platzhalter, deaktiviert).
- Keine automatische/LLM-Bewertung (existiert bereits in `agent/verifier.py`).
- Kein Recall / „verpasste Detektionen" — nur Bewertung des Erkannten (Precision).

## Architektur

Im Ordner `src/litter_detection/visualisation/dashboard/`:

1. **`panels/validation_logic.py`** — reine, testbare Logik (keine Gradio-Abhängigkeit):
   - `Verdict` — Konstanten `CORRECT` (`"müll"`), `INCORRECT` (`"kein_müll"`), `UNSURE` (`"unsicher"`).
   - `detection_key(detection) -> str` — stabiler Schlüssel aus `timestamp|label|position`.
   - `compute_stats(keys, verdicts) -> ValidationStats` — Dataclass mit
     `total, correct, incorrect, unsure, pending, precision` (`precision` = `correct /
     (correct + incorrect)` als Prozent-Float, oder `None`, wenn Nenner 0).
   - `next_pending_key(ordered_keys, verdicts, after_key=None) -> str | None` — nächster
     unbewerteter Schlüssel nach `after_key` (zyklisch von vorne), sonst `None`.

2. **`panels/validation.py`** — `ValidationPanel` (gleiches Muster wie die anderen Panels):
   - Hält Verdikte in einem Instanz-`dict[str, str]` (nur im Speicher, prozess-global —
     für die Einzelnutzer-UI ausreichend in diesem Schritt).
   - `render()` baut die Komponenten und gibt sie als Dataclass zurück.
   - Update-/Handler-Methoden für Auswahl und Bewertung; nutzt `validation_logic`.
   - Datenquelle: `provider.get_trash_detections()` (dieselben Detektionen wie das
     Trash-Panel).

3. **`visualization.py`** — das bestehende Dashboard wird in `gr.Tabs` gefasst:
   - **Tab „Betrieb"**: die bisherige Cockpit-Ansicht, unverändert.
   - **Tab „Validierung"**: rendert das `ValidationPanel`.
   - CSS-Ergänzungen für den Validierungs-Tab im bestehenden Blau/Lila/Gold-SCADA-Stil.

## UI-Aufbau Tab „Validierung"

- **Kennzahl-Leiste:** Gesamt · Korrekt · Fehlalarm · Unsicher · Offen · **Precision %**.
- **Aktuelle Detektion:** großes Bild + Metadaten (Label, Confidence, Zeit, Position).
- **Bewertungs-Buttons:** ✓ Müll (Blau) · ✗ Kein Müll (Rot) · ? Unsicher (Gold).
  Ein Klick setzt das Verdikt und springt zur nächsten **offenen** Detektion.
- **Liste/Galerie** aller Detektionen mit Verdikt-Markierung; Filter „Alle / Nur offene".
- **Export-Button:** sichtbar, aber deaktiviert (Platzhalter für den Persistenz-Schritt).

Der Validierungs-Tab darf scrollen (Review-Werkzeug); die No-Scroll-Regel gilt nur für
den Betrieb-Tab.

## Datenfluss

```
Provider.get_trash_detections()  ->  Liste TrashDetection
        |                                   |
        v                                   v
   detection_key(d)               Anzeige (Bild + Metadaten + Galerie)
        |
   Klick ✓/✗/?  ->  verdicts[key] = Verdict.X
        |
   compute_stats(keys, verdicts)  ->  Kennzahl-Leiste aktualisieren
   next_pending_key(...)          ->  nächste offene Detektion anzeigen
```

## Fehlerbehandlung

- Keine Detektionen vorhanden → leere, aber valide Ansicht (Hinweistext, Stats = 0).
- Verdikt für nicht mehr vorhandene Detektion → wird in `compute_stats` ignoriert
  (Stats basieren auf den aktuell vorhandenen Keys).

## Tests

`tests/test_validation_logic.py` (stdlib `unittest`, kein Zusatz-Dependency):

- `compute_stats`: leere Eingabe; gemischte Verdikte; Precision-Berechnung; Precision
  `None`, wenn nur „unsicher"/„offen" (kein korrekt/falsch).
- `next_pending_key`: erste offene; nach `after_key`; zyklischer Umlauf; `None` wenn alle
  bewertet.
- `detection_key`: stabil und unterscheidet verschiedene Detektionen.

Die Gradio-Render-Schicht wird nicht unit-getestet; ihre Korrektheit wird durch
erfolgreichen `build_dashboard`-Aufbau und manuelle Sichtprüfung im Browser verifiziert.
