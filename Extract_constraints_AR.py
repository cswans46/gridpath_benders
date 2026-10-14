import mosek as msk
from datetime import datetime
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import math
import sys, random
from pyomo.core.kernel.constraint import constraint


hasup = lambda bk: bk in [msk.boundkey.up, msk.boundkey.ra, msk.boundkey.fx]
haslo = lambda bk: bk in [msk.boundkey.lo, msk.boundkey.ra, msk.boundkey.fx]
relaxup = lambda bk: msk.boundkey.fr if bk == msk.boundkey.up else msk.boundkey.lo
relaxlo = lambda bk: msk.boundkey.fr if bk == msk.boundkey.lo else msk.boundkey.up

sgn = lambda x: '-' if x < 0.0 else '+'
varname = lambda t, j: t.getvarname(j) if t.getvarnamelen(j) > 0 else f"x[{j}]"
conname = lambda t, i: f"{t.getconname(i)}: \n" if t.getconnamelen(i) > 0 else ""
btoineq = lambda b, bl, bu: f"\t <= {bu}" if b == 'u' else f"\t >= {bl}"


# Prepare a task for the deletion filter routine
def prepareTask(task):
    # Use the simplex algporithm to exploit hot-start
    task.putintparam(msk.iparam.optimizer, msk.optimizertype.free_simplex)
    # Allow mixed-integer models by working with their continuous relaxation instead
    task.putintparam(msk.iparam.mio_mode, msk.miomode.ignored)
    # Remove objective
    task.putclist(range(task.getnumvar()), [0.0] * task.getnumvar())


# Check if a task is feasible
def feasibilityStatus(task):
    task.optimize()
    psta = task.getprosta(msk.soltype.bas)
    if psta in [msk.prosta.prim_and_dual_feas]:
        return True
    elif psta in [msk.prosta.prim_infeas]:
        return False
    else:
        return None  # Could be numerical issues


def get_random_constraint_list(mps_task_filepath):
    output_file = mps_task_filepath.parent / "random_items.txt"
    allItems = []
    with msk.Task() as task:
        print("start reading")
        task.readdataformat(filename=str(mps_task_filepath), format=msk.dataformat.task, compress=msk.compresstype.zstd)
        prepareTask(task)
        print("start computing")
        # Find all essential (not free) bounds in the problem
        # Format: (index, constraint or variable ? , lower or upper bound ? )
        allItems = [(i, 'c', 'u') for i in range(task.getnumcon()) if hasup(task.getconbound(i)[0])] + \
                   [(i, 'c', 'l') for i in range(task.getnumcon()) if haslo(task.getconbound(i)[0])] + \
                   [(j, 'v', 'u') for j in range(task.getnumvar()) if hasup(task.getvarbound(j)[0])] + \
                   [(j, 'v', 'l') for j in range(task.getnumvar()) if haslo(task.getvarbound(j)[0])]

        print("Number of constraints and variables: ", len(allItems))

    with open(output_file, "w") as file:
        for (idx, what, bound) in allItems:
            if random.uniform(0, 1) < 0.99:
                if what == 'v':
                    bk, bl, bu = task.getvarbound(idx)
                    file.write(f"+ {varname(task, idx)}{btoineq(bound, bl, bu)}\n")
                else:
                    bk, bl, bu = task.getconbound(idx)
                    nz, sub, val = task.getarow(idx)
                    expr = ' '.join(
                        f"\t\t{sgn(v)} {abs(v)} {varname(task, j)}\n" for (j, v) in zip(list(sub), list(val)))
                    file.write(f"{conname(task, idx)}{expr}{btoineq(bound, bl, bu)}\n")


def check_feasability(mps_task_filepath):
    with msk.Task() as task:
        print("start reading")
        task.readdataformat(filename=str(mps_task_filepath), format=msk.dataformat.task, compress=msk.compresstype.zstd)
        prepareTask(task)
        if feasibilityStatus(task) != False:
            print("The task is feasible")
        else:
            print("The task is not feasible")
    return


if __name__ == "__main__":

    # python mosek_iis.py --mps_filepath "D:\data\scn_data_artax\testing\obfuscation\scenarios\BAU_12days_US_LowNGPrice_CGII_240917_grouped_prov_and_CM\logs\mosek_dump_file_binary_symbolic.task.zst"
    parser = ArgumentParser()
    parser.add_argument(
        "--mps_filepath",
        help="MPS filename",
        type=str,
    )

    parsed_args = parser.parse_args()

    mps_task_filepath = Path(parsed_args.mps_filepath)

    # will save a txt file with 1% of the constraints and variable (at random).
    get_random_constraint_list(mps_task_filepath)
