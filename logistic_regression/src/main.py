import os
import aux.colors as c
import preprocessing
import correlations
import describe
import normalize

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    input(f"\n{c.YELLOW}Press ENTER to Continue...{c.RESET}")

def show_menu():
    text_menu = f"""
    {c.PURPLE}===========   M E N U   ===========
    {c.CYAN}1.{c.BLUE} Preprocesing data
    {c.CYAN}2.{c.BLUE} Remove highly correlated columns
    {c.CYAN}3.{c.BLUE} Describe dataset (max, min, ...)
    {c.CYAN}4.{c.BLUE} Normalize dataset
    {c.YELLOW}\t   ----  BONUS  ----
    {c.CYAN}5.{c.BLUE} Option 5
    {c.CYAN}6.{c.BLUE} Option 6
    {c.CYAN}7.{c.BLUE} Option 7
    {c.CYAN}0. Exit
    {c.PURPLE}{'='*36}{c.RESET}
    """
    print(text_menu)

def execute_option(choice):
    clear_screen()
    if choice == 0:
        print(f"{c.PURPLE}Bye! 👋{c.RESET}\n")
        return False
    elif choice == 1:
        print(f"{c.CYAN}=== Preprocessing Data ==={c.RESET}\n")
        preprocessing.preprocess_data()
    elif choice == 2:
        print(f"{c.CYAN}=== Removing Highly Correlated Columns ==={c.RESET}\n")
        correlations.remove_highly_correlated_columns()
    elif choice == 3:
        print(f"{c.CYAN}=== Analyzing Dataset ==={c.RESET}\n")
        describe.analyze_dataset()
    elif choice == 4:
        print(f"{c.CYAN}=== Normalizing Dataset ==={c.RESET}\n")
        normalize.normalize_data()
    else:
        print(f"{c.RED}Invalid Option{c.RESET}")
    
    pause()
    return True

if __name__ == "__main__":
    running = True
    while running:
        clear_screen()
        show_menu()
        
        try:
            choice = int(input(f"{c.GREEN}Insert Option: {c.RESET}"))
            running = execute_option(choice)
        except ValueError:
            print(f"{c.RED}Value out of range. Please try again.{c.RESET}")
            pause()