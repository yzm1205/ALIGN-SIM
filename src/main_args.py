from argparse import ArgumentParser


def get_args():
    """
    Parses command-line arguments for SentencePerturbation.
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--cf", 
        dest="csv_file", 
        required=True, 
        help="Name of the CSV file"
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        choices=["anto", "jumbling", "syn", "paraphrase"],
        help="Task to perform: anto/jumbling/syn/paraphrase",
    )
    parser.add_argument(
        "--M", 
        dest="model_name", 
        required=True, 
        help="LLM Model")
    
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Save the results in a CSV file",
    )
    
    parser.add_argument(
        "--gpu", 
        dest="gpu", 
        default="auto", 
        help="GPU to run the model"
    )
    
    parser.add_argument(
        "--batch_size", 
        dest="batch_size", 
        type=int, 
        default=16, 
        help="Batch size for translation"
    )
    
    parser.add_argument(
        "--metric",
        dest="metric",
        type=str,
        default="cosine",
        choices=["cosine","ned","both"],
        help="Metric to use for comparison",
    )
    return parser.parse_args()
