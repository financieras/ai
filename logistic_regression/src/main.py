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
import prepare_test
import predict_test

# Define dataset path as constant
TRAIN_DATASET = '../datasets/dataset_train.csv'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    input(f"\n{c.YELLOW}Press ENTER to Continue...{c.RESET}")

def show_menu():
    text_menu = f"""
    {c.PURPLE}=========================  M E N U  =========================
    {c.CYAN}  1.{c.BLUE} Initial data analysis             [initial_analysis.py]
    {c.CYAN}  2.{c.BLUE} Correlation analysis              [correlations.py]
    {c.CYAN}  3.{c.BLUE} Preprocessing data                [preprocessing.py]
    {c.CYAN}  4.{c.BLUE} Describe dataset                  [describe.py]
    {c.CYAN}  5.{c.BLUE} Histogram                         [histogram.py]
    {c.CYAN}  6.{c.BLUE} Scatter Plot                      [scatter_plot.py]
    {c.CYAN}  7.{c.BLUE} Pair Plot                         [pair_plot.py]
    {c.CYAN}  8.{c.BLUE} Normalize                         [normalize.py]
    {c.CYAN}  9.{c.BLUE} Train                             [logreg_train.py]
    {c.CYAN} 10.{c.BLUE} Predict                           [logreg_predict.py]
    {c.CYAN} 11.{c.BLUE} Prepare test dataset              [prepare_test.py]
    {c.CYAN} 12.{c.BLUE} Predict using test data           [predict_test.py]
    {c.CYAN}  0. Exit
    {c.PURPLE}{'='*61}{c.RESET}
    """
    print(text_menu)

def execute_option(choice_input):
    clear_screen()
    
    try:
        choice = int(choice_input.strip())
    except ValueError:
        print(f"{c.RED}Invalid input. Please enter a number.{c.RESET}")
        return True

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
    elif choice == 11:
        print(f"{c.CYAN}=== Preparing and normlizing TEST dataset ==={c.RESET}\n")
        prepare_test.main()
    elif choice == 12:
        print(f"{c.CYAN}=== Predict using TEST dataset and training weights ==={c.RESET}\n")
        predict_test.main()
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
            choice_input = input(f"{c.GREEN}Insert Option: {c.RESET}")
            running = execute_option(choice_input)
        except ValueError:
            print(f"{c.RED}Value out of range. Please try again.{c.RESET}")
            pause()