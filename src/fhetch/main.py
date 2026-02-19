from argparse import ArgumentParser
from pathlib import Path

from .env import default_global
from .eval import eval_main, eval_globals
from .parser import Program
from .validate import validate_all
from .sugar import desugar


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("program", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    prog_path = args.program
    prog = Program.parse_file(prog_path, parse_all=True).program
    #desugar(prog)
    global_env = default_global()
    global_env.update(eval_globals(prog))
    validate_all(prog, global_env)
    eval_main(prog, global_env)


if __name__ == "__main__":
    main()
