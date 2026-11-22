#!/usr/bin/env python3
"""Run a coding mission with an explicit approval step."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex.agents.workspace import Workspace
from cortex.coding.agent import CodingAction, CodingAgent, ScriptedPolicy
from cortex.memory.store import MemoryStore
from cortex.planning.missions import MissionStore


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Workspace(Path(directory))
        workspace.write("README.md", "# Projet de démonstration\n")
        store = MemoryStore(Path(directory) / "missions.sqlite")
        missions = MissionStore(store)
        policy = ScriptedPolicy([
            CodingAction("list_files", {}),
            CodingAction("write_file", {"path": "hello.py", "content": "print('bonjour')\n"}),
        ])
        agent = CodingAgent(workspace, missions, policy, auto_approve=False)
        mission = agent.start_mission("Créer hello.py", "Créer un script hello.py")
        print("1.", agent.run_step(mission.id))
        print("2.", agent.run_step(mission.id))
        print("3. approbation :", agent.approve(mission.id))
        print("Fichier créé :", workspace.read("hello.py").strip())
        print("Mission :", missions.get(mission.id).status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
