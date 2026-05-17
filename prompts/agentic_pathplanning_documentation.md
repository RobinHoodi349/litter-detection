# Entwicklung eines PathPlanner- und Explore-Agenten für die autonome Robodog-Flächensuche

## Prompt:

Du bist ein erfahrener KI-Engineer für agentenbasierte Robotiksysteme. Ich arbeite an einem Robodog-Projekt zur autonomen Müllsuche auf einem Hochschulcampus. Es soll ein Multi-Agent-System entstehen, in dem ein Robodog ein vordefiniertes rechteckiges Suchfeld systematisch abläuft.

**Aktuelles Problem:**  
Es existiert noch keine klare Agentenlogik für die Pfadplanung und Exploration. Der Roboter soll nicht manuell gesteuert werden, sondern vom Coordination Agent eine Suchmission erhalten. Der Explore Agent soll diese Mission ausführen, der PathPlanner Agent soll dafür einen systematischen Coverage Path berechnen und der Move Agent soll die Bewegung des Roboters übernehmen.

**Gewünschte Änderungen / Anforderungen:**

1. Es soll ein `PathPlannerAgent` erstellt werden, der auf eine `PLAN_PATH`-Anfrage reagiert.
2. Der `PathPlannerAgent` soll die aktuelle Roboterposition über ein Zenoh-Topic abfragen.
3. Ausgehend von der aktuellen Roboterposition soll ein rechteckiges Suchfeld berechnet werden.
4. Die Feldgröße (`width_m`, `height_m`) und der Spurabstand (`lane_spacing_m`) sollen vom Explore Agent übergeben werden.
5. Der Pfad soll als Boustrophedon-/Lawnmower-Pattern erzeugt werden.
6. Die Ausgabe des PathPlanner Agents soll eine strukturierte Waypoint-Liste enthalten.
7. Es soll ein `ExploreAgent` erstellt werden, der vom Coordination Agent eine `START_EXPLORATION`-Anfrage erhält.
8. Der Explore Agent soll die Missionsparameter an den PathPlanner Agent weitergeben.
9. Nach Erhalt der Waypoints soll der Explore Agent diese sequenziell an den bestehenden `MoveAgent` übergeben.
10. Der Explore Agent soll auf `STOP_EXPLORATION`, `BLOCK` und `UNBLOCK` reagieren können.
11. Der `MoveAgent` soll nicht neu implementiert werden, sondern nur über eine saubere Schnittstelle mit Zielkoordinaten angesprochen werden.
12. Die Umsetzung soll modular bleiben und zur bestehenden Projektstruktur passen.

Erstelle die finalen Versionen von `pathPlanerAgent.py` und `exploreAgent.py`. Erkläre anschließend kurz, welche Rolle beide Agents im Multi-Agent-System übernehmen und warum der PathPlanner Agent regelbasiert statt LLM-basiert umgesetzt wird.

## Auswertung

[Ausführliche Beschreibung der Metric]

| Metric                              | Score         |
|-------------------------------------|---------------|
| **Tool used**                       | ChatGPT       |
| **Error Rate (0 - 4)**              | 4             |
| **Code Quality (0 - 4)**            | 4             |
| **Discrepancy from Prompt (0 - 4)** | 4             |
| **Notes**                           | PathPlannerAgent mit Zenoh-basierter Positionsabfrage erstellt; Boustrophedon-/Lawnmower-Pfadplanung umgesetzt; ExploreAgent als Orchestrierungsagent implementiert; Übergabe der Waypoints an bestehenden MoveAgent vorbereitet; BLOCK, UNBLOCK und STOP_EXPLORATION berücksichtigt |