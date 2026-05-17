# Dynamisches Replanning für Robodog-Müllerkennung

## Prompt:

Sie sind ein erfahrener Robotics- und KI-System-Engineer mit fundiertem Wissen in Multi-Agent-Systemen, autonomer Exploration, Frontier-based Path Planning und robotischer Missionskoordination. Ich habe drei Agent-Dateien für ein KI-basiertes Robodog-System zur autonomen Müllsuche auf einem Hochschulcampus, die ich überarbeiten möchte:

- `coordinator.py`
- `exploreAgent.py`
- `pathPlanerAgent.py`

**Aktuelles Problem:**  
Das Zusammenspiel zwischen `PathPlannerAgent`, `ExploreAgent` und `Coordinator Agent` ist nicht korrekt. Der `PathPlannerAgent` erzeugt aktuell nur einmalig beim Start der Mission einen statischen Coverage-Pfad. Dieser Pfad wird anschließend vom `ExploreAgent` linear abgefahren. Neue Informationen aus der Umgebung, z. B. neue Frontiers, Hindernisse, SLAM-Updates oder bestätigte Müllfunde, führen nicht automatisch zu einem neuen Fahrplan.

Das ist für einen autonomen Robodog auf einem Hochschulcampus ungeeignet, da sich die Umweltinformationen während der Exploration kontinuierlich ändern. Der Roboter soll nicht starr einem alten Pfad folgen, sondern seine Route dynamisch an neue Informationen anpassen.

**Gewünschte Änderungen an den drei Agent-Dateien:**

1. Der `Coordinator Agent` soll als zentrale Steuerungseinheit den aktuellen World State verwalten.
2. Sobald neue Informationen eintreffen, z. B. neue Frontiers, Hindernisse oder bestätigte Müllfunde, soll der `Coordinator Agent` ein Replanning auslösen.
3. Der `PathPlannerAgent` soll nicht nur einmalig einen Coverage-Path erzeugen, sondern bei jedem Replanning-Aufruf anhand der aktuellen Roboterposition und des aktualisierten World State einen neuen Fahrplan berechnen.
4. Der `ExploreAgent` soll einen laufenden Fahrplan dynamisch ersetzen können, ohne dass die gesamte Mission neu gestartet werden muss.
5. Bestehendes Verhalten wie Start, Stop, Block und Unblock der Exploration soll erhalten bleiben.
6. Die Änderungen sollen möglichst minimalinvasiv sein, damit die bestehende Projektstruktur und vorhandene Schnittstellen nicht unnötig verändert werden.
7. Das System soll weiterhin funktionieren, wenn keine Frontier-Daten vorhanden sind. In diesem Fall soll der bisherige Coverage-Path als Fallback verwendet werden.
8. Die Kommunikation soll weiterhin über die vorhandene Zenoh-basierte Architektur passen.
9. Die Lösung soll Thread-Sicherheit berücksichtigen, da Exploration, Detector und Replanning parallel laufen können.

**Zusätzliche Frage:**  
Falls es eine bessere Architektur für das Zusammenspiel zwischen Wahrnehmung, World State, Pfadplanung und Bewegung gibt, erkläre diese kurz und integriere sie gegebenenfalls in die Lösung.

Schreibe die vollständig überarbeiteten Versionen von:

- `coordinator.py`
- `exploreAgent.py`
- `pathPlanerAgent.py`

Trenne die drei Dateien sauber voneinander, sodass sie direkt per Copy-Paste übernommen werden können. Erkläre anschließend kurz, welche Änderungen in jeder Datei vorgenommen wurden.

## Auswertung

[Ausführliche Beschreibung der Metric]

| Metric                              | Score         |
|-------------------------------------|---------------|
| **Tool used**                       | ChatGPT       |
| **Error Rate (0 - 4)**              | 4             |
| **Code Quality (0 - 4)**            | 4             |
| **Discrepancy from Prompt (0 - 4)** | 4             |
| **Notes**                           | Dynamisches Replanning umgesetzt; Coordinator verwaltet World State; ExploreAgent unterstützt REPLACE_PLAN; PathPlanner nutzt Frontier-Daten mit Coverage-Fallback |