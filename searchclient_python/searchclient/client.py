"""
Entry point del cliente multiagente.

Reemplaza el searchclient.py del warmup (que hacía búsqueda en espacio de
joint actions — O(b^n_agents), no escalable).

Nueva arquitectura:
  - LevelParser.parse() → State inicial + LevelProfile
  - Manager.setup()     → agentes, distancias, asignación de tareas
  - Loop turno a turno: Manager.get_joint_action() → enviar al servidor

Ejecutar:
    python -m searchclient.client -level <level_file>
  o via el servidor MAvis:
    java -jar server.jar -l <level> -c "python -m searchclient.client"
"""

from __future__ import annotations

import argparse
import sys

from searchclient import memory
from searchclient.level_parser import LevelParser
from searchclient.manager import Manager


def main(args: argparse.Namespace) -> None:
    print("MultiAgentClient initializing.", file=sys.stderr, flush=True)

    # Send client name to server (protocol requirement).
    # First stdout line = client name; lines starting with '#' are comments.
    print("MultiAgentClient", flush=True)
    print("#Multi-agent hospital client 02285 F26", flush=True)

    server_messages = sys.stdin

    # --- CAPA 1: Parse level ---
    initial_state, profile = LevelParser.parse(server_messages)
    print(
        f"Parsed level: {profile.num_rows}x{profile.num_cols}, "
        f"{profile.num_agents} agents, "
        f"{len(profile.real_boxes)} real boxes, "
        f"{len(profile.deco_boxes)} deco boxes.",
        file=sys.stderr,
        flush=True,
    )

    # --- CAPA 2+3+4: Setup manager ---
    manager = Manager()
    manager.setup(initial_state, profile)

    # --- Execution loop ---
    current_state = initial_state
    t = 0
    MAX_STEPS = 19_500  # stay safely inside 20,000 limit

    while not manager.is_done(current_state) and t < MAX_STEPS:
        joint_action = manager.get_joint_action(current_state)

        # Send joint action to server (one line: "act1|act2|...")
        print(
            "|".join(a.name_ for a in joint_action),
            flush=True,
        )

        # Read server response (must drain stdin to avoid blocking server)
        response = server_messages.readline()
        if not response:
            break

        # Advance local state mirror (always apply — manager already resolved conflicts)
        current_state = current_state.result(joint_action)

        t += 1
        if t % 500 == 0:
            print(
                f"t={t}, memory={memory.get_usage():.1f}/{memory.max_usage:.0f} MB",
                file=sys.stderr,
                flush=True,
            )

    if manager.is_done(current_state):
        print(f"Solved at t={t}.", file=sys.stderr, flush=True)
    else:
        print(f"Did not solve within {t} steps.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent hospital client.")
    parser.add_argument(
        "--max-memory",
        metavar="<MB>",
        type=float,
        default=10048.0,
        help="Soft memory limit in MB.",
    )
    parser.add_argument(
        "--bypass-debugger",
        action="store_true",
        help="Bypass debugger wait (for local testing without debugpy).",
    )
    args = parser.parse_args()
    memory.max_usage = args.max_memory

    if not args.bypass_debugger:
        import debugpy

        debugpy.listen(("localhost", 1234))
        debugpy.wait_for_client()
    main(args)
