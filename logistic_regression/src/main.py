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

# Define dataset paths as constants
TRAIN_DATASET = '../datasets/dataset_train.csv'
TEST_DATASET = '../datasets/dataset_test.csv'

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

    Note: Add 'test' after the number to use test dataset (e.g., '1 test')
    {c.PURPLE}{'='*65}{c.RESET}
    """
    print(text_menu)

def execute_option(choice_input):
    clear_screen()
    
    # Parse input to check for test dataset option
    parts = choice_input.strip().split()
    choice = parts[0]
    use_test = len(parts) > 1 and parts[1].lower() == 'test'
    
    try:
        choice = int(choice)
    except ValueError:
        print(f"{c.RED}Invalid input. Please enter a number or 'number test'{c.RESET}")
        return True

    if choice == 0:
        print(f"{c.PURPLE}Bye! 👋{c.RESET}\n")
        return False
    elif choice == 1:
        print(f"{c.CYAN}=== Initial data analysis ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        initial_analysis.initial_exploration(input_file=input_file)
    elif choice == 2:
        print(f"{c.CYAN}=== Detecting Highly Correlated Columns ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        correlations.detect_highly_correlated_columns(input_file=input_file)
    elif choice == 3:
        print(f"{c.CYAN}=== Preprocessing Data ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        preprocessing.preprocess_data(input_file=input_file)
    elif choice == 4:
        print(f"{c.CYAN}=== Analyzing Dataset ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        describe.analyze_dataset(input_file=input_file)
    elif choice == 5:
        print(f"{c.CYAN}=== Making the histogram (wait a moment...) ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        histogram.generate_histogram(input_file=input_file)
    elif choice == 6:
        print(f"{c.CYAN}=== Making the Scatter Plot (wait a moment...) ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        scatter_plot.generate_scatter_plot(input_file=input_file)
    elif choice == 7:
        print(f"{c.CYAN}=== Making the Pair Plot (wait a moment...) ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        pair_plot.exploring_feature_relationships(input_file=input_file)
    elif choice == 8:
        print(f"{c.CYAN}=== Normalizing Dataset ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        normalize.normalize_data(input_file=input_file)
    elif choice == 9:
        print(f"{c.CYAN}=== Model training ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        logreg_train.main(input_file=input_file)
    elif choice == 10:
        print(f"{c.CYAN}=== Predict the Model ==={c.RESET}\n")
        input_file = TEST_DATASET if use_test else TRAIN_DATASET
        logreg_predict.main(input_file=input_file)
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