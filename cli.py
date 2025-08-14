import argparse
from config import APPLICATION_NAME, APPLICATION_DESCRIPTION


def get_arguments():
    parser = argparse.ArgumentParser(prog=APPLICATION_NAME,
                                     description=APPLICATION_DESCRIPTION)
    
    parser.add_argument("-i","--input", type=str, required=True, dest="input_image")

    return parser.parse_args()

def main():
    arguments = get_arguments()
    print(arguments.input_image)
    
if __name__ == '__main__':
    main()