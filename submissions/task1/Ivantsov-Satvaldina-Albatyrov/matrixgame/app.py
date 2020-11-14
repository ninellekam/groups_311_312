import sys
import argparse
import importlib


def import_mod(mod):
    try:
        module = importlib.import_module(f'.{mod}', 'matrixgame.commands')    
    except ModuleNotFoundError:
        print('Invalid command')
        sys.exit(1)

    getattr(module, 'Command')().run()


def main():
    parser = argparse.ArgumentParser(description='Matrixgame Project')
    parser.add_argument('-e','--equilibrium', help='visualizes nash equilibrium in pure strategies',
                        action='store_const', const='equilibrium')
    parser.add_argument('-i','--incomplete', help='visualizes that the optimal strategy spectrum is incomplete',
                         action='store_const', const='incomplete')
    parser.add_argument('-c','--complete', help='visualizes that the optimal strategy spectrum is complete',
                         action='store_const', const='complete')

    args = parser.parse_args()
    if args.equilibrium:
        import_mod(args.equilibrium)
    elif args.incomplete:
        import_mod(args.incomplete)
    elif args.complete:
        import_mod(args.complete)






