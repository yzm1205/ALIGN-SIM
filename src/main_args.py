from argparse import ArgumentParser


def get_args():
    """
    Parses command-line arguments for SentencePerturbation.
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", 
        dest="dataset", 
        choices=["qqp","mrpc","paws","negation"],
        required=True, 
        help="Name of the CSV file"
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        default="all",
        nargs="+",
        choices=["anto", "antonym", "jumbling","jumb" ,"syn","synonym", "paraphrase","negation", "all"],
        help="Task(s) to perform: anto/jumbling/syn/paraphrase/all. Can specify multiple tasks.",
    )
    parser.add_argument(
        "--model", 
        dest="model_name", 
        required=False, 
        help="LLM Model")
    
    parser.add_argument(
        "--target_lang",
        dest="target_lang",
        default="en",
        help="Language for translation"
    )
    
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Save the results in a CSV file",
    )
    
    parser.add_argument(
        "--gpu", 
        dest="gpu", 
        default="cuda:0", 
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
    
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        default=3500,
        type=int,
        help="Number of rows to process.",
    )

    parser.add_argument(
        "--custom-class-path",
        dest="custom_class_path",
        default=None,
        help="Import path for a custom embedder implementing BaseEmbedder.",
    )

    parser.add_argument(
        "--custom-kwargs",
        dest="custom_kwargs",
        default=None,
        help="JSON string of keyword arguments passed to the custom embedder.",
    )

    args = parser.parse_args()
    if args.model_name is None and args.custom_class_path is None:
        parser.error("Either --model or --custom-class-path must be provided.")

    return args
