import os
import aux.colors as c
import initial_analysis
import preprocessing
import correlations
import describe
import normalize
import histogram
import scatter_plot
import pair_plot
import logreg_train
import logreg_predict

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    input(f"\n{c.YELLOW}Press ENTER to Continue...{c.RESET}")

def show_menu():
    text_menu = f"""
    {c.PURPLE}=========================    M E N U    =========================
    {c.CYAN}  1.{c.BLUE} Initial data analysis                 [initial_analysis.py]
    {c.CYAN}  2.{c.BLUE} Detect highly correlated columns      [correlations.py]
    {c.CYAN}  3.{c.BLUE} Preprocessing data                    [preprocessing.py]
    {c.CYAN}  4.{c.BLUE} Describe dataset (max, min, ...)      [describe.py]
    {c.CYAN}  5.{c.BLUE} Histogram                             [histogram.py]
    {c.CYAN}  6.{c.BLUE} Scatter Plot                          [scatter_plot.py]
    {c.CYAN}  7.{c.BLUE} Pair Plot                             [pair_plot.py]
    {c.CYAN}  8.{c.BLUE} Normalize dataset                     [normalize.py]
    {c.CYAN}  9.{c.BLUE} Train                                 [logreg_train.py]
    {c.CYAN} 10.{c.BLUE} Predict                               [logreg_predict.py]
    {c.CYAN}  0. Exit
    {c.PURPLE}{'='*65}{c.RESET}
    """
    print(text_menu)

def execute_option(choice):
    clear_screen()
    if choice == 0:
        print(f"{c.PURPLE}Bye! 👋{c.RESET}\n")
        return False
    elif choice == 1:
        print(f"{c.CYAN}=== Initial data analysis ==={c.RESET}\n")
        initial_analysis.initial_exploration()
    elif choice == 2:
        print(f"{c.CYAN}=== Detecting Highly Correlated Columns ==={c.RESET}\n")
        correlations.detect_highly_correlated_columns()
    elif choice == 3:
        print(f"{c.CYAN}=== Preprocessing Data ==={c.RESET}\n")
        preprocessing.preprocess_data()
    elif choice == 4:
        print(f"{c.CYAN}=== Analyzing Dataset ==={c.RESET}\n")
        describe.analyze_dataset()
    elif choice == 5:
        print(f"{c.CYAN}=== Making the histogram (wait a moment...) ==={c.RESET}\n")
        histogram.generate_histogram()
    elif choice == 6:
        print(f"{c.CYAN}=== Making the Scatter Plot (wait a moment...) ==={c.RESET}\n")
        scatter_plot.generate_scatter_plot()
    elif choice == 7:
        print(f"{c.CYAN}=== Making the Pair Plot (wait a moment...) ==={c.RESET}\n")
        pair_plot.exploring_feature_relationships()
    elif choice == 8:
        print(f"{c.CYAN}=== Normalizing Dataset ==={c.RESET}\n")
        normalize.normalize_data()
    elif choice == 9:
        print(f"{c.CYAN}=== Model training ==={c.RESET}\n")
        logreg_train.main()
    elif choice == 10:
        print(f"{c.CYAN}=== Predict the Model ==={c.RESET}\n")
        logreg_predict.main()
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