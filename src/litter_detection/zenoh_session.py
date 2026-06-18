"""Zentrales Öffnen von Zenoh-Sessions.

Alle Prozesse dieses Stacks sollen mit *genau einem* Router reden (dem in
``router`` angegebenen) und niemals Traffic zwischen Routern weiterleiten.

Deshalb wird die Session im **client**-Modus geöffnet und das Multicast-/Gossip-
Scouting deaktiviert. Im Default-Modus ("peer") würde ein Prozess zusätzlich per
Multicast den Router eines *zweiten* Robodogs entdecken, sich mit ihm verbinden
und beide Netze überbrücken — dann erreichen Kommandos beide Roboter. Client-Modus
verhindert genau dieses Weiterrouten.
"""
import json

import zenoh


def open_zenoh_session(router: str | None) -> zenoh.Session:
    """Öffnet eine Zenoh-Session im client-Modus, verbunden mit genau einem Router."""
    conf = zenoh.Config()
    conf.insert_json5("mode", json.dumps("client"))
    if router:
        conf.insert_json5("connect/endpoints", json.dumps([router]))
    conf.insert_json5("scouting/multicast/enabled", "false")
    conf.insert_json5("scouting/gossip/enabled", "false")
    return zenoh.open(conf)
