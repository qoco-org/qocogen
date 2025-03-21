import os
import struct


def run_generated_solver(solver_dir):
    results = ""
    if os.name == 'nt':
        results = "/build/Debug/result.bin"
        os.system(
            "cd "
            + solver_dir
            + " && mkdir build && cd build && cmake -DQOCO_CUSTOM_BUILD_TYPE:STR=Release .. && cmake --build . && cd Debug && .\\runtest.exe && cd ../.."
        )
    else:
        results = "/build/result.bin"
        os.system(
            "cd "
            + solver_dir
            + " && mkdir build && cd build && cmake -DQOCO_CUSTOM_BUILD_TYPE:STR=Release .. && cmake --build . && ./runtest && cd ../.."
        )
    with open(solver_dir + results, "rb") as file:
        # Read the unsigned int (4 bytes)
        solved = struct.unpack("B", file.read(1))[0]

        # Read the first double (8 bytes)
        obj = struct.unpack("d", file.read(8))[0]

        # Read the second double (8 bytes)
        average_runtime_ms = struct.unpack("d", file.read(8))[0]
    return solved, obj, average_runtime_ms
