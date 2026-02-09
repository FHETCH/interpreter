from argparse import ArgumentParser
from pathlib import Path

from .parser import Program
from .validate import validate_all
from .eval import eval_main, eval_consts
from .sugar import desugar


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("program", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    prog_path = args.program
    prog = Program.parse_file(prog_path, parse_all=True).program
    eval_consts(prog)
    desugar(prog)
    validate_all(prog)
    eval_main(prog)
