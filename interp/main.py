from argparse import ArgumentParser
from pathlib import Path

from interp.parser import Program
from interp.validate import validate_all
from interp.eval import eval_main, eval_consts
from interp.sugar import desugar


def main(prog_path):
    prog = Program.parse_file(prog_path, parse_all=True).program
    eval_consts(prog)
    desugar(prog)
    validate_all(prog)
    # print(prog)
    eval_main(prog)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    main(args.program)
