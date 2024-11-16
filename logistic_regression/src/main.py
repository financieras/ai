import os
import aux.colors as c
import preprocessing
import correlations
import describe
import normalize

def clear_screen():
    # Use `os.system('cls')` to Windows or `os.system('clear')` to Linux/macOS
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    text_menu = f"{c.PURPLE}" + \
    """
    ============   M E N U   ============
    1. Preprocesing data
    2. Remove highly correlated columns
    3. Describe dataset (max, min, ...)
    4. Normalize dataset
    0. Exit
    =====================================
    """ + \
    f"{c.RESET}"
    print(text_menu)



if __name__ == "__main__":
    while True:
        clear_screen()
        show_menu()
        
        try:
            choice = int(input("Insert Option: "))
            if choice == 0:
                print(f"{c.PURPLE}Bye! 👋{c.RESET}\n")
                exit()
            elif choice == 1:
                preprocessing.preprocess_data()
            elif choice == 2:
                correlations.remove_highly_correlated_columns()
            elif choice == 3:
                describe.analyze_dataset()
            elif choice == 4:
                normalize.normalize_data()
            else:
                print("Opción inválida")
        except ValueError:
            print("Value out of range. Please try again.")
