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
    text_menu = f"""
    {c.PURPLE}===========   M E N U   ===========
    {c.CYAN}1.{c.BLUE} Preprocesing data
    {c.CYAN}2.{c.BLUE} Remove highly correlated columns
    {c.CYAN}3.{c.BLUE} Describe dataset (max, min, ...)
    {c.CYAN}4.{c.BLUE} Normalize dataset
    {c.YELLOW}\t   ----  BONUS  ----
    {c.CYAN}5.{c.BLUE} Opción 5
    {c.CYAN}6.{c.BLUE} Opción 6
    {c.CYAN}7.{c.BLUE} Opción 7
    {c.CYAN}0. Exit
    {c.PURPLE}{'='*36}{c.RESET}
    """
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
