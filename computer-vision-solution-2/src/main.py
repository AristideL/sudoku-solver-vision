import argparse
from .gui_app import App


def main():
    ap = argparse.ArgumentParser(description='Sudoku CV Solver - Solution 2')
    ap.add_argument('--debug_dir', default=None, help='If set, save debug artifacts per run under this directory')
    args = ap.parse_args()

    App(debug_dir=args.debug_dir).mainloop()


if __name__ == '__main__':
    main()
