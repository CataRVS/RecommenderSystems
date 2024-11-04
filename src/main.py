import sys
from src.data import Data

def menu():
    """
    Display the menu. The options to choose from are:
    1. Load data from file
    """
    print('Menu:')
    print('1. Load data from file')
    print('2. Exit')
    incorrect_choice = True
    choice = input('Enter your choice: ').strip()
    while incorrect_choice:
        if choice == '1':
            data = ask_data_path()
            incorrect_choice = False
            return data
        elif choice == '2':
            sys.exit()
        else:
            print('Incorrect choice. Please try again.')
            choice = input('Enter your choice: ')


def ask_data_path():
    """
    Ask the user for the path to the dataset.
    """
    data_path = input('Enter the path to the dataset: ')
    incorrect_path = True
    while incorrect_path:
        try:
            data = Data(data_path)
            incorrect_path = False
            return data
        except FileNotFoundError:
            print('File not found. Please try again.')
            data_path = input('Enter the path to the dataset: ')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            data = Data(sys.argv[1])
        except FileNotFoundError:
            print('File not found. Please try again.')
    else:
        data = menu()
    print(data.get_total_items())
    print(data.get_total_users())
    print(len(data.get_mappings()))
    print(data.get_data())
    print(data.get_data()[0].head())
    print(data.get_data()[1].head())
    print(data.get_user_ratings(49))
